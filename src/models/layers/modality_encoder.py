import pdb
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.layer import BasicBlock
from einops import rearrange
import pickle
import math
from models.wavlm.WavLM import WavLM, WavLMConfig
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config, RobertaModel, RobertaTokenizer
from typing import Optional
from models.layers.transformer import Mlp
from models.wav2vec2.model import wav2vec2_model

class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=2, target_length=256):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1700, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
        
        self.length_adjuster = ExactLengthAdjuster(target_length=target_length)
    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        
        out = self.length_adjuster(out)
        
        return out.transpose(1, 2)


class ExactLengthAdjuster(nn.Module):
    """
    Layer that ensures the output has exactly the target length along the time dimension.
    It either adds or removes frames as needed.
    """
    def __init__(self, target_length=196):
        super(ExactLengthAdjuster, self).__init__()
        self.target_length = target_length
    
    def forward(self, x):
        # x is expected to be [batch, channels, time]
        current_length = x.shape[2]
        
        if current_length == self.target_length:
            return x
        elif current_length < self.target_length:
            # Need to add frames
            frames_to_add = self.target_length - current_length
            
            # Duplicate the last frame as many times as needed
            last_frame = x[:, :, -1:]
            extra_frames = last_frame.repeat(1, 1, frames_to_add)
            
            return torch.cat([x, extra_frames], dim=2)
        else:
            # Need to remove frames
            # Just truncate to the target length
            return x[:, :, :self.target_length]


class ModalityEncoder(nn.Module):
    def __init__(self, 
                 data_path, 
                 t_fix_pre, 
                 audio_dim, 
                 audio_in=2,
                 raw_audio=False,
                 latent_dim=256,
                 audio_fps=30,
                 use_exp=False,
                 target_length=256,
                 ):
        super().__init__()
        
        self.raw_audio = raw_audio
        self.latent_dim = latent_dim
        self.audio_fps = audio_fps
        

        self.WavEncoder = WavEncoder(audio_dim, audio_in=audio_in, target_length=target_length)
        self.text_encoder_body = nn.Linear(300, audio_dim) 

        with open(f"{data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=t_fix_pre)
        word_dim = pre_trained_embedding.shape[1]

        if use_exp:
            self.mix_audio_text = nn.Linear(audio_dim*2, self.latent_dim)
        else:
            self.mix_audio_text = nn.Linear(audio_dim*2, self.latent_dim)
    
    def forward(self, in_audio=None, cached_audio_low=None, cached_audio_high=None, cached_intention=None, cached_intention_mask=None, audio=None, word=None, squeeze_scale=4):
        # Initial features extraction - single transpose each
        # [B, T, D] -> [T, B, D]
        audio_feat = self.WavEncoder(audio)
        text_feat = self.text_encoder_body(self.text_pre_encoder_body(word))
        
        
        at_feat = torch.cat([audio_feat, text_feat], dim=2)  # [B, T, D]
        
        at_feat = self.mix_audio_text(at_feat)  # [B, T, D']
        
        at_feat = F.avg_pool1d(at_feat.transpose(1, 2), squeeze_scale)
        at_feat = at_feat.transpose(1, 2) # [B, T/scale, D']
        return {
            "audio_low":at_feat,
            "audio_high":None,
            }
    

class IntentionalAudioEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_size = 768
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=self.args.hidden_size,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.args.n_layer)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        
        self.ffn = nn.Sequential(
            nn.Linear(768, self.args.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.args.hidden_size, 768)
        )

        self.intention_pad_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.xavier_uniform_(self.intention_pad_token)

    def forward(self, audio_feat, intention_feat, intention_mask):
        pad_mask = ~intention_mask  # True for pad, False for valid
        all_pad = pad_mask.all(dim=1)  # [batch]
        if all_pad.any():
            # Replace with learnable pad token
            intention_feat[all_pad] = self.intention_pad_token.expand(all_pad.sum(), intention_feat.size(1), intention_feat.size(2))
            pad_mask[all_pad, 0] = False  # Unmask first token

        intention_feat = self.encoder(intention_feat, src_key_padding_mask=pad_mask)
        audio_norm = self.norm1(audio_feat)
        intention_norm = self.norm2(intention_feat)
        attn_output, _ = self.cross_attention(
            query=audio_norm,
            key=intention_norm,
            value=intention_norm,
            key_padding_mask=pad_mask
        )
        audio_feat = audio_feat + attn_output
        ffn_output = self.ffn(self.norm3(audio_feat))
        audio_feat = audio_feat + ffn_output
        return audio_feat


class WrapedWav2Vec(nn.Module):
    def __init__(self, args):
        super(WrapedWav2Vec, self).__init__()
        self.args = args
        self.feature_extractor = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').feature_extractor
        self.feature_projection = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').feature_projection
        self.encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').encoder
        
        self.encoder.layers = self.encoder.layers[:self.args.n_layer]
        
        self.proj_down = nn.Linear(768,512)
        
        
        self.intent_encode = getattr(self.args, "intent_encode", False)
        if self.intent_encode:
            self.intent_encoder = IntentionalAudioEncoder(self.args)
            
    
    def forward(self, 
        inputs,
        intention_feat: Optional[torch.Tensor] = None,
        intention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        separate_levels: bool = True
        ):
        finetune_audio_low = self.feature_extractor(inputs).transpose(1, 2)
        hidden_states, _ = self.feature_projection(finetune_audio_low.detach() if separate_levels else finetune_audio_low)
        
        if self.intent_encode and intention_feat is not None:
            intention_feat = self.intent_encoder(hidden_states)
            hidden_states = self.intent_encoder(hidden_states, intention_feat, intention_mask)
        
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        hidden_states = self.proj_down(hidden_states)
        
        
        return {
            "low_level": finetune_audio_low,
            "high_level": hidden_states
        }



class WarpedDistillWav2Vec(nn.Module):
    def __init__(self, args):
        super(WarpedDistillWav2Vec, self).__init__()
        self.args = args
        
        pretrained_model_name = './datasets/DPWavLM-sp0.75.pth'
        ckpt = torch.load(pretrained_model_name)
        wav2vec2_encoder = wav2vec2_model(**ckpt["config"])
        wav2vec2_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        
        self.feature_extractor = wav2vec2_encoder.feature_extractor
        self.feature_projection = wav2vec2_encoder.encoder.feature_projection
        self.encoder = wav2vec2_encoder.encoder.transformer
        self.encoder.layers = self.encoder.layers[:self.args.n_layer]
        
        self.proj_down = nn.Linear(768,512)
        
        
        self.intent_encode = getattr(self.args, "intent_encode", False)
        if self.intent_encode:
            self.intent_encoder = IntentionalAudioEncoder(self.args)
        

    def forward(self, 
        inputs,
        intention_feat: Optional[torch.Tensor] = None,
        intention_mask: Optional[torch.Tensor] = None,
        separate_levels: bool = True
        ):
        finetune_audio_low, _ = self.feature_extractor(inputs, length=None)
        hidden_states = self.feature_projection(finetune_audio_low.detach() if separate_levels else finetune_audio_low,)
        
        if self.intent_encode and intention_feat is not None:
            hidden_states = self.intent_encoder(hidden_states, intention_feat, intention_mask)
        
        encoder_outputs = self.encoder(hidden_states,)
        hidden_states = encoder_outputs
        hidden_states = self.proj_down(hidden_states)
        
        return {
            "low_level": finetune_audio_low,
            "high_level": hidden_states
        }




class MultiModalEncoder(nn.Module): # the bert model is freezed for now
    def __init__(self, args):
        super(MultiModalEncoder, self).__init__()
        self.args = args
        
        # Initialize modality encoders
        
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        
        if getattr(self.args, "use_distill", False):
            self.audio_encoder_fintune = WarpedDistillWav2Vec(self.args)
        else:
            self.audio_encoder_fintune = WrapedWav2Vec(self.args) 
        
        self.audio_high_mapping = Mlp(512+512+512, self.args.hidden_size, self.args.audio_f)
        
        if getattr(self.args, "use_distill", False):
            self.audio_low_mapping = Mlp(176+176, self.args.hidden_size, self.args.audio_f)
        else:
            self.audio_low_mapping = Mlp(512+512, self.args.hidden_size, self.args.audio_f)
        
        self.audio_down_proj_2 = nn.Linear(768, 512)
        self.audio_down_proj_3 = nn.Linear(768, 512)
        
        self.semantic_fusion = getattr(self.args, "semantic_fusion", False)
        # use a cross-attention to fuse the intention with bert_time output
        if self.semantic_fusion:
            self.semantic_fusion_layer = IntentionalAudioEncoder(self.args)
        
        self.finetune_bert = getattr(self.args, "finetune_bert", False) # default is False, not going to implement this to be True to save memory and time
        if self.finetune_bert:
            self.bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert_model = RobertaModel.from_pretrained('roberta-base')
            self.asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        
        self.down_sample = 4
        
        # freeze the audio encoder
        for param in self.audio_encoder_fintune.parameters():
            param.requires_grad = False
    
    def load_from_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pretrained weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys in state_dict match
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract the encoder's state dict from the full model state dict
        state_dict = checkpoint['model_state_dict']
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('audio_encoder_fintune.'):
                new_key = key[len('audio_encoder_fintune.'):]
                encoder_state_dict[new_key] = value
        
        missing_keys, unexpected_keys = self.audio_encoder_fintune.load_state_dict(encoder_state_dict, strict=strict)
        
        if strict:
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
        return checkpoint.get('iteration', None)  # Return the iteration number if available
    
    def forward(self, in_audio=None, cached_audio_low=None, cached_audio_high=None, cached_intention=None, cached_intention_mask=None, audio_onset=None, word=None):
        

        if cached_audio_low is not None:
            raw_audio_low = cached_audio_low
            raw_audio_high_rhythmic = cached_audio_high[:, :, :768]
            raw_audio_high_semantic = cached_audio_high[:, :, 768:]
            
            if self.semantic_fusion:
                raw_audio_high_semantic = self.semantic_fusion_layer(raw_audio_high_semantic, cached_intention, cached_intention_mask)
            
            raw_audio_high_rhythmic = self.audio_down_proj_2(raw_audio_high_rhythmic)
            raw_audio_high_semantic = self.audio_down_proj_3(raw_audio_high_semantic)

            raw_audio_high = torch.cat([raw_audio_high_rhythmic, raw_audio_high_semantic], dim=-1)
            
            audio_list = [i.cpu().numpy() for i in in_audio]
            inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(in_audio.device)
            
            finetune_audio = self.audio_encoder_fintune(inputs.input_values, intention_feat=cached_intention, intention_mask=cached_intention_mask, separate_levels=False)
            finetune_audio_low, finetune_audio_high = finetune_audio["low_level"], finetune_audio["high_level"]
            
            
            diff = raw_audio_high.shape[1] - finetune_audio_high.shape[1]
            if diff > 0:
                finetune_audio_high = torch.cat([finetune_audio_high, finetune_audio_high[:, -diff:]], dim=1)
            diff = raw_audio_low.shape[1] - finetune_audio_low.shape[1]
            if diff > 0:
                finetune_audio_low = torch.cat([finetune_audio_low, finetune_audio_low[:, -diff:]], dim=1)
            
            
            raw_audio_low = torch.cat([raw_audio_low, finetune_audio_low], dim=-1) # bs, t, 1024
        else:
            print("error! must have cached audio in training")
        
        tar_size_1 = round(raw_audio_low.shape[1] / 50 * 30 / self.down_sample)
        tar_size_2 = round(raw_audio_high.shape[1] / 50 * 30 / self.down_sample)
        
        raw_audio_low = F.interpolate(raw_audio_low.transpose(1, 2), size=tar_size_1, mode='linear', align_corners=True).transpose(1, 2) 
        
        raw_audio_high = F.interpolate(raw_audio_high.transpose(1, 2), size=tar_size_2, mode='linear', align_corners=True).transpose(1, 2)
        finetune_audio_high = F.interpolate(
            finetune_audio_high.transpose(1, 2), size=tar_size_2, mode='linear', align_corners=True
        ).transpose(1, 2)  
        
        audio_low = self.audio_low_mapping(raw_audio_low)
        raw_audio_high = torch.cat([finetune_audio_high, raw_audio_high], dim=-1)
        
        audio_high = self.audio_high_mapping(raw_audio_high)
        
        
        return {
            "audio_low":audio_low,
            "audio_high":audio_high,
            }



class MultiModalEncoderwithWavCNN(nn.Module): # the bert model is freezed for now
    def __init__(self, args):
        super(MultiModalEncoderwithWavCNN, self).__init__()
        self.args = args
        
        # Initialize modality encoders
        
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        
        if getattr(self.args, "use_distill", False):
            self.audio_encoder_fintune = WarpedDistillWav2Vec(self.args)
        else:
            self.audio_encoder_fintune = WrapedWav2Vec(self.args) 
        
        self.audio_high_mapping = Mlp(512+512+512, self.args.hidden_size, self.args.audio_f)
        
        self.audio_down_proj_2 = nn.Linear(768, 512)
        self.audio_down_proj_3 = nn.Linear(768, 512)
        
        self.semantic_fusion = getattr(self.args, "semantic_fusion", False)
        # use a cross-attention to fuse the intention with bert_time output
        if self.semantic_fusion:
            self.semantic_fusion_layer = IntentionalAudioEncoder(self.args)
        
        self.finetune_bert = getattr(self.args, "finetune_bert", False) # default is False, not going to implement this to be True to save memory and time
        if self.finetune_bert:
            self.bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert_model = RobertaModel.from_pretrained('roberta-base')
            self.asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        
        self.down_sample = 4
        
        # freeze the audio encoder
        for param in self.audio_encoder_fintune.parameters():
            param.requires_grad = False
        if self.semantic_fusion:
            for param in self.semantic_fusion_layer.parameters():
                param.requires_grad = False
    
    
    
        # WavCNN
        self.WavEncoder = WavEncoder(512, audio_in=2, target_length=256)
        self.text_encoder_body = nn.Linear(300, 512) 

        with open(f"./datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=False)
        word_dim = pre_trained_embedding.shape[1]

        self.mix_audio_text = nn.Linear(512*2, 512)
    

    
    def load_from_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pretrained weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys in state_dict match
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract the encoder's state dict from the full model state dict
        state_dict = checkpoint['model_state_dict']
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('audio_encoder_fintune.'):
                new_key = key[len('audio_encoder_fintune.'):]
                encoder_state_dict[new_key] = value
        
        missing_keys, unexpected_keys = self.audio_encoder_fintune.load_state_dict(encoder_state_dict, strict=strict)
        
        if strict:
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
        return checkpoint.get('iteration', None)  # Return the iteration number if available
    
    def forward(self, in_audio=None, cached_audio_low=None, cached_audio_high=None, cached_intention=None, cached_intention_mask=None, audio_onset=None, word=None):
        
        
        # Low-level audio feature extraction
        # [B, T, D] -> [T, B, D]
        audio_feat = self.WavEncoder(audio_onset)
        text_feat = self.text_encoder_body(self.text_pre_encoder_body(word))
        
        
        at_feat = torch.cat([audio_feat, text_feat], dim=2)  # [B, T, D]
        
        at_feat = self.mix_audio_text(at_feat)  # [B, T, D']
        
        at_feat = F.avg_pool1d(at_feat.transpose(1, 2), 4)
        audio_low = at_feat.transpose(1, 2) # [B, T/scale, D']
        

        if cached_audio_high is not None:
            raw_audio_high_rhythmic = cached_audio_high[:, :, :768]
            raw_audio_high_semantic = cached_audio_high[:, :, 768:]
            
            if self.semantic_fusion:
                raw_audio_high_semantic = self.semantic_fusion_layer(raw_audio_high_semantic, cached_intention, cached_intention_mask)
            
            raw_audio_high_rhythmic = self.audio_down_proj_2(raw_audio_high_rhythmic)
            raw_audio_high_semantic = self.audio_down_proj_3(raw_audio_high_semantic)

            raw_audio_high = torch.cat([raw_audio_high_rhythmic, raw_audio_high_semantic], dim=-1)
            
            audio_list = [i.cpu().numpy() for i in in_audio]
            inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(in_audio.device)
            
            finetune_audio = self.audio_encoder_fintune(inputs.input_values, intention_feat=cached_intention, intention_mask=cached_intention_mask, separate_levels=False)
            finetune_audio_low, finetune_audio_high = finetune_audio["low_level"], finetune_audio["high_level"]
            
            
            diff = raw_audio_high.shape[1] - finetune_audio_high.shape[1]
            if diff > 0:
                finetune_audio_high = torch.cat([finetune_audio_high, finetune_audio_high[:, -diff:]], dim=1)
            
        else:
            print("error! must have cached audio in training")
        
        tar_size_2 = round(raw_audio_high.shape[1] / 50 * 30 / self.down_sample)
        
        
        raw_audio_high = F.interpolate(raw_audio_high.transpose(1, 2), size=tar_size_2, mode='linear', align_corners=True).transpose(1, 2)
        finetune_audio_high = F.interpolate(
            finetune_audio_high.transpose(1, 2), size=tar_size_2, mode='linear', align_corners=True
        ).transpose(1, 2)  
        
        raw_audio_high = torch.cat([finetune_audio_high, raw_audio_high], dim=-1)
        
        audio_high = self.audio_high_mapping(raw_audio_high)
        
        
        return {
            "audio_low":audio_low,
            "audio_high":audio_high,
            }



class BaselineEncoder(nn.Module): # the bert model is freezed for now
    def __init__(self, args):
        super(BaselineEncoder, self).__init__()
        self.args = args
        
        # Initialize modality encoders
        
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        
        if getattr(self.args, "use_distill", False):
            self.audio_encoder_fintune = WarpedDistillWav2Vec(self.args)
        else:
            self.audio_encoder_fintune = WrapedWav2Vec(self.args) 
        
        self.audio_high_mapping = Mlp(512+512+512, self.args.hidden_size, self.args.audio_f)
        
        if getattr(self.args, "use_distill", False):
            self.audio_low_mapping = Mlp(176+176, self.args.hidden_size, self.args.audio_f)
        else:
            self.audio_low_mapping = Mlp(512+512, self.args.hidden_size, self.args.audio_f)
        
        self.audio_down_proj_2 = nn.Linear(768, 512)
        self.audio_down_proj_3 = nn.Linear(768, 512)
        
        self.semantic_fusion = getattr(self.args, "semantic_fusion", False)
        # use a cross-attention to fuse the intention with bert_time output
        if self.semantic_fusion:
            self.semantic_fusion_layer = IntentionalAudioEncoder(self.args)
        
        self.finetune_bert = getattr(self.args, "finetune_bert", False) # default is False, not going to implement this to be True to save memory and time
        if self.finetune_bert:
            self.bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert_model = RobertaModel.from_pretrained('roberta-base')
            self.asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        
        self.down_sample = 4
    
    def load_from_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pretrained weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys in state_dict match
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract the encoder's state dict from the full model state dict
        state_dict = checkpoint['model_state_dict']
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('audio_encoder_fintune.'):
                new_key = key[len('audio_encoder_fintune.'):]
                encoder_state_dict[new_key] = value
        
        missing_keys, unexpected_keys = self.audio_encoder_fintune.load_state_dict(encoder_state_dict, strict=strict)
        
        if strict:
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
        return checkpoint.get('iteration', None)  # Return the iteration number if available
    
    def forward(self, in_audio=None, cached_audio_low=None, cached_audio_high=None, cached_intention=None, cached_intention_mask=None):
        

        if cached_audio_low is not None:
            raw_audio_low = cached_audio_low
            raw_audio_high_rhythmic = cached_audio_high[:, :, :768]
            raw_audio_high_semantic = cached_audio_high[:, :, 768:]
            
            if self.semantic_fusion:
                raw_audio_high_semantic = self.semantic_fusion_layer(raw_audio_high_semantic, cached_intention, cached_intention_mask)
            
            raw_audio_high_rhythmic = self.audio_down_proj_2(raw_audio_high_rhythmic)
            raw_audio_high_semantic = self.audio_down_proj_3(raw_audio_high_semantic)

            raw_audio_high = torch.cat([raw_audio_high_rhythmic, raw_audio_high_semantic], dim=-1)
            
            audio_list = [i.cpu().numpy() for i in in_audio]
            inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(in_audio.device)
            
            finetune_audio = self.audio_encoder_fintune(inputs.input_values, intention_feat=cached_intention, intention_mask=cached_intention_mask, separate_levels=False)
            finetune_audio_low, finetune_audio_high = finetune_audio["low_level"], finetune_audio["high_level"]
            
            diff = raw_audio_high.shape[1] - finetune_audio_high.shape[1]
            if diff > 0:
                finetune_audio_high = torch.cat([finetune_audio_high, finetune_audio_high[:, -diff:]], dim=1)
            diff = raw_audio_low.shape[1] - finetune_audio_low.shape[1]
            if diff > 0:
                finetune_audio_low = torch.cat([finetune_audio_low, finetune_audio_low[:, -diff:]], dim=1)
            
            raw_audio_low = torch.cat([raw_audio_low, finetune_audio_low], dim=-1) # bs, t, 1024
        else:
            print("error! must have cached audio in training")
        
        tar_size_1 = math.ceil(raw_audio_low.shape[1] / 50 * 30 / self.down_sample)
        tar_size_2 = math.ceil(raw_audio_high.shape[1] / 50 * 30 / self.down_sample)
        breakpoint()
        raw_audio_low = F.interpolate(raw_audio_low.transpose(1, 2), size=tar_size_1, mode='linear', align_corners=True).transpose(1, 2) 
        
        raw_audio_high = F.interpolate(raw_audio_high.transpose(1, 2), size=tar_size_2, mode='linear', align_corners=True).transpose(1, 2)
        finetune_audio_high = F.interpolate(
            finetune_audio_high.transpose(1, 2), size=tar_size_2, mode='linear', align_corners=True
        ).transpose(1, 2)  
        
        audio_low = self.audio_low_mapping(raw_audio_low)
        raw_audio_high = torch.cat([finetune_audio_high, raw_audio_high], dim=-1)
        
        audio_high = self.audio_high_mapping(raw_audio_high)
        
        
        return {
            "audio_low":audio_high,
            "audio_high":None,
            }



class AudioCNNEncoder(nn.Module):
    def __init__(self, 
                 audio_dim=512, 
                 audio_in=2,
                 latent_dim=256,
                 ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.WavEncoder = WavEncoder(audio_dim, audio_in=audio_in)

        self.projection = nn.Linear(audio_dim, latent_dim)

        
    
    def forward(self,  in_audio=None, cached_audio_low=None, cached_audio_high=None, cached_intention=None, cached_intention_mask=None, audio_onset=None, squeeze_scale=4):
        # Initial features extraction - single transpose each
        # [B, T, D] -> [T, B, D]
        audio_feat = self.WavEncoder(audio_onset)
        
        
        at_feat = self.projection(audio_feat)  # [B, T, D']
        
        at_feat = F.avg_pool1d(at_feat.transpose(1, 2), squeeze_scale)
        at_feat = at_feat.transpose(1, 2) # [B, T/scale, D']
        return {
            "audio_low":at_feat,
            "audio_high":None,
            }