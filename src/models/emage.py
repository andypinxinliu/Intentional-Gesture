import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .layers.utils import *
from .layers.transformer import Mlp
from .layers.gpt import TransformerBlock, ModelArgs, find_multiple, KVCache


class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_codebooks):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.ModuleList()
        for _ in range(num_codebooks):
            codebook = nn.Embedding(vocab_size // num_codebooks, hidden_size)
            self.codebooks.append(codebook)

    def forward(self, indices):
        # indices: [B, K, L]
        assert indices.shape[1] == self.num_codebooks
        latent_features = []
        for i in range(self.num_codebooks):
            latent_feature = self.codebooks[i](indices[:, i]) # [B, L, D]
            latent_features.append(latent_feature)
        latent_features = torch.stack(latent_features).sum(dim=0) # [K, B, L, D] -> [B, L, D]
        return latent_features

class AutoRegressiveHead(nn.Module):
    def __init__(self, num_codebooks, vocab_size, dim, n_output_layer):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        self.codebooks = nn.ModuleList()
        for _ in range(self.num_codebooks - 1):
            codebook = nn.Embedding(self.vocab_size, dim)
            self.codebooks.append(codebook)

        self.args = ModelArgs(dim=dim, n_layer=n_output_layer, n_head=4, attn_dropout_p=0.0, resid_dropout_p=0.1)
        self.layers = torch.nn.ModuleList()
        for _ in range(n_output_layer):
            self.layers.append(TransformerBlock(self.args, drop_path=0.))

        self.norm = nn.LayerNorm(dim)
        self.linear_head = nn.Linear(dim, self.vocab_size)

    def forward_train(self, base_tokens, targets):
        K = targets.shape[1]
        B, L, C = base_tokens.shape
        base_tokens = base_tokens.reshape(B * L, 1, C)
        targets = targets.permute(0, 2, 1).reshape(B * L, K)[:, :-1]
        index_embeddings = []
        for i in range(self.num_codebooks - 1):
            index_embed = self.codebooks[i](targets[:, i])
            index_embeddings.append(index_embed)
        index_embeddings = torch.stack(index_embeddings, dim=1)
        h = torch.cat((base_tokens, index_embeddings), dim=1) # [B*L, K, D]
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=None, mask=None)
        h = self.norm(h)
        logits = self.linear_head(h)
        logits = logits.reshape(B, L, K, -1).permute(0, 2, 1, 3) # [B, L, K, V]
        return logits

    def forward_test(self, base_tokens, idx=None, input_pos=None, mask=None):
        
        if idx is not None:
            # idx [B, L]
            B, L = idx.shape
            idx = idx.reshape(B * L, 1)
            h = self.codebooks[input_pos - 1](idx)
        else:
            h = base_tokens
            # [B, L, D]
            B, L, D = h.shape
            h = h.reshape(B * L, 1, D)
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        logits = self.linear_head(h)
        logits = logits.reshape(B, L, -1)
        return logits
    
    def setup_head_caches(self, max_batch_size, max_seq_length, dtype, device):
        head_dim = self.args.dim // self.args.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype).to(
                device)
        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).to(device)
        self.head_causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)

class EMAGE(nn.Module):
    def __init__(self,
        input_dim=128,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        n_seed=8,
        seq_len=32,
        flip_sin_to_cos= True,
        freq_shift = 0,
        cond_proj_dim=None,
        use_exp=False,
        num_speakers=1,
        audio_dim=512,
        embed_context_multiplier=4,
    ):
        super().__init__()
        
       
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.use_exp = use_exp
        self.joint_num = 3 if not self.use_exp else 4
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.audio_dim = audio_dim
        self.embed_context_multiplier = embed_context_multiplier
        
        self.condition_proj = None
        self.intent_proj = None
        if self.audio_dim != self.latent_dim:
            self.condition_proj = nn.Linear(self.audio_dim, self.latent_dim)
            self.intent_proj = nn.Linear(self.audio_dim, self.latent_dim)

        
        if num_speakers > 1:
            self.embed_id = nn.Embedding(30, self.latent_dim) # actually only 25 is used, the rest is padding, i am lazy to change it
        else:
            self.embed_id = None

        
        self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_heads)
        
        
        self.mask_embedding = nn.Parameter(torch.zeros(1,1,self.latent_dim), requires_grad=True)
        nn.init.normal_(self.mask_embedding, 0, self.latent_dim**-0.5)

        self.input_process = nn.Linear(self.latent_dim, self.latent_dim)
        
        
        ## EMAGE definition, modifed from https://github.com/PantoMatrix/PantoMatrix/blob/main/models/emage_audio/modeling_emage_audio.py
        
        self.token_embedder = TokenEmbedder(vocab_size=8192, hidden_size=self.input_dim, num_codebooks=8)

        # motion encoder
        self.audio_body_motion_proj = nn.Linear(self.audio_dim, self.latent_dim)
        self.moton_proj = nn.Linear(self.latent_dim, self.latent_dim)
        
        
        self.transformer_en_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=4, dim_feedforward=ff_size)
        self.motion_self_encoder = nn.TransformerEncoder(self.transformer_en_layer, num_layers=1)
        # coss attn
        self.audio_motion_cross_attn_layer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=4, dim_feedforward=ff_size)
        self.audio_motion_cross_attn = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=8)
        
        # feed forward
        self.motion2latent = Mlp(self.latent_dim, ff_size, self.latent_dim)
        
        # decoder
        self.body_motion_decoder = AutoRegressiveHead(num_codebooks=8, vocab_size=8192, dim=self.latent_dim, n_output_layer=4)
        
            
    def forward(self, at_feat, speaker_id, motion_idx, mask, use_audio=True):
        
        masked_motion = self.token_embedder(motion_idx) # [B, L, D]
        
        # mask motion
        masked_embeddings = self.mask_embedding.expand_as(masked_motion)
        masked_motion = torch.where(mask==1, masked_embeddings, masked_motion)

        # motion token (spatial hints)
        body_hint = self.input_process(masked_motion)
        body_hint_body = self.bodyhints_body(body_hint)

        speaker_motion_fea_proj = self.embed_id(speaker_id)
        

        # motion self attn (temporal)
        masked_motion_proj = self.moton_proj(body_hint_body)
        masked_motion_proj = self.position_embeddings(masked_motion_proj)
        masked_motion_proj = speaker_motion_fea_proj + masked_motion_proj
        motion_fea = self.motion_self_encoder(masked_motion_proj.permute(1,0,2)).permute(1,0,2)

        # audio_cross_attn
        
        audio_proj = self.audio_body_motion_proj(at_feat)
        
        motion_fea = motion_fea + speaker_motion_fea_proj
        motion_fea = self.position_embeddings(motion_fea)
        audio_cross = self.audio_motion_cross_attn(tgt=motion_fea.permute(1,0,2), memory=audio_proj.permute(1,0,2)).permute(1,0,2)
        if not use_audio:
          audio_cross = audio_cross * 0.
        motion_fea = motion_fea + audio_cross 

        # mlp
        latent = self.motion2latent(motion_fea + speaker_motion_fea_proj)
        
        motion_logits = None
        # decode
        if self.training:
            motion_logits = self.body_motion_decoder.forward_train(latent, targets=motion_idx)
        
        return {
            "logits": motion_logits,
            "latent": latent,
        }
    

    def inference(self, at_feat, speaker_id, motion_idx, mask, use_audio=True, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True):
        # step 1; generate the base token
        base_token = self.forward(at_feat, speaker_id, motion_idx, mask, use_audio=True)['latent'] # [B, L, D]
        
        self.body_motion_decoder.setup_head_caches(max_batch_size=1, max_seq_len=8, dtype=base_token.dtype)
        # step 2: based on the base token, autoregressively generate the next codebook token
        for i in range(self.body_motion_decoder.num_codebooks):
            start_pos = torch.tensor([i], dtype=torch.int)
            mask = None
            if i == 0:
                logits = self.body_motion_decoder.forward_test(base_token, input_pos=start_pos, mask=mask)
            else:
                logits = self.body_motion_decoder.forward_test(base_token, idx=pred_idx, input_pos=start_pos, mask=mask)
            pred_idx = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, sample_logits=sample_logits)[0] # [B, L]
            indices.append(pred_idx)
        indices = torch.stack(indices, dim=1) # [B, K, L]
        return indices # [B, K, L]