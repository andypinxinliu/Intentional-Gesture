import copy
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import difflib
from typing import Optional, Tuple, Union

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config, RobertaModel, RobertaTokenizer
from models.layers.motion_encoder import WrapedMotionEncoder, WrapedMotionEncoderV2
from models.layers.modality_encoder import WrapedWav2Vec, WarpedDistillWav2Vec, IntentionalAudioEncoder
from models.layers.transformer import Mlp
from models.utils.audio_utils import audio_to_time_aligned_text_features



class JointEmbedding(nn.Module):
    def __init__(self, args):
        super(JointEmbedding, self).__init__()
        self.args = args.model
        
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

        if getattr(self.args, "use_distill", False):
            self.audio_encoder_fintune = WarpedDistillWav2Vec(self.args)
        else:
            self.audio_encoder_fintune = WrapedWav2Vec(self.args)
        
        self.finetune_bert = getattr(self.args, "finetune_bert", False) # default is False, not going to implement this to be True to save memory and time
        
        if self.finetune_bert:
            self.bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert_model = RobertaModel.from_pretrained('roberta-base')
            self.asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

        if getattr(self.args, "use_distill", False):
            self.audio_low_mapping = Mlp(176+176, self.args.hidden_size, self.args.audio_f)
        else:
            self.audio_low_mapping = Mlp(512+512, self.args.hidden_size, self.args.audio_f)
            
        self.audio_high_mapping = Mlp(512+512+512, self.args.hidden_size, self.args.audio_f)
        
        self.audio_down_proj_2 = nn.Linear(768, 512)
        self.audio_down_proj_3 = nn.Linear(768, 512)
        
        if getattr(self.args, "v2", False):
            self.motion_encoder_fintune = WrapedMotionEncoderV2(self.args)
        else:
            self.motion_encoder_fintune = WrapedMotionEncoder(self.args)
        
        self.motion_low_mapping = Mlp(self.args.motion_f, self.args.hidden_size, self.args.motion_f)
        self.motion_high_mapping = Mlp(self.args.motion_f, self.args.hidden_size, self.args.motion_f)
        
        self.semantic_fusion = getattr(self.args, "semantic_fusion", False)
        # use a cross-attention to fuse the intention with bert_time output
        if self.semantic_fusion:
            self.semantic_fusion_layer = IntentionalAudioEncoder(self.args)
        
        
        self.down_sample = 4 # for downsample 30 fps motion to 7.5 (the same as audio and vq downsample)
        self.smplx_model = None
        self.get_motion_reps = None
        self.audio_to_time_aligned_text_features = audio_to_time_aligned_text_features
        self.low_temp = nn.Parameter(torch.tensor(0.07))
        self.low_level_loss_fn = None
        self.high_temp = nn.Parameter(torch.tensor(0.07))
        self.high_level_loss_fn = None

    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.args.hidden_size ** -0.5)
    
    def forward(self, in_audio=None, in_motion=None, cached_audio_low=None, cached_audio_high=None, cached_rep15d=None, cached_intention=None, cached_intention_mask=None):
        # motion feature
        if cached_rep15d is not None:
            in_motion = cached_rep15d
        else:
            in_motion = self.get_motion_reps(in_motion, self.smplx_model)["rep15d"] # now it is 30fps
        
        motion_features = self.motion_encoder_fintune(in_motion)
        raw_motion_low = motion_features["low_level"] # self.motion_encoder_low(in_motion)
        raw_motion_high = motion_features["high_level"] # self.motion_encoder_high(torch.cat([raw_motion_low.detach(), in_motion], dim=-1))

        motion_low = self.motion_low_mapping(raw_motion_low)
        motion_high = self.motion_high_mapping(raw_motion_high)

       
        motion_cls = motion_high.mean(dim=1)

        # audio feature
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
            finetune_audio = self.audio_encoder_fintune(inputs.input_values, intention_feat=cached_intention, intention_mask=cached_intention_mask)
            finetune_audio_low, finetune_audio_high = finetune_audio["low_level"], finetune_audio["high_level"]
            
            raw_audio_low = torch.cat([raw_audio_low, finetune_audio_low], dim=-1) # bs, t, 1024
        else:
            print("error! must have cached audio in training")
        
        tar_size_1 = math.ceil(raw_audio_low.shape[1] / 50 * 30 / self.down_sample)
        tar_size_2 = math.ceil(raw_audio_high.shape[1] / 50 * 30 / self.down_sample)
        
        
        raw_audio_low = F.interpolate(raw_audio_low.transpose(1, 2), size=tar_size_1, mode='linear', align_corners=True).transpose(1, 2) 
        
        raw_audio_high = F.interpolate(raw_audio_high.transpose(1, 2), size=tar_size_2, mode='linear', align_corners=True).transpose(1, 2)
        
        finetune_audio_high = F.interpolate(
            finetune_audio_high.transpose(1, 2), size=tar_size_2, mode='linear', align_corners=True
        ).transpose(1, 2)  
        
        audio_low = self.audio_low_mapping(raw_audio_low)
        audio_high = torch.cat([finetune_audio_high, raw_audio_high], dim=-1)
        
        audio_high = self.audio_high_mapping(audio_high)
        
        audio_cls = audio_high.mean(dim=1)
        
        # fix temp to 0.1 is better than learned temp
        low_infonce, low_acc = self.low_level_loss_fn(audio_low, motion_low)
        high_infonce = self.high_level_loss_fn(audio_cls, motion_cls)
        
        return {
            "audio_low":audio_low,
            "audio_high":audio_high,
            "audio_cls":audio_cls,
            "motion_low":motion_low,
            "motion_high":motion_high,
            "motion_cls":motion_cls,
            "low_level_loss": [low_infonce, low_acc],
            "high_level_loss": high_infonce
            }