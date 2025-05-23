import math
import copy
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class AdaLayerNormShift(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        shift = self.linear(self.silu(emb.to(torch.float32)).to(emb.dtype))
        x = self.norm(x) + shift.unsqueeze(dim=1)
        return x


class IntentionJointAttention(Attention):
    def __init__(self, added_intention_dim, **kwargs):
        super().__init__(**kwargs)
        self.add_intent_k_proj = nn.Linear(added_intention_dim, self.inner_kv_dim, bias=self.added_proj_bias)
        self.add_intent_v_proj = nn.Linear(added_intention_dim, self.inner_kv_dim, bias=self.added_proj_bias)
        
        if self.context_pre_only is not None:
            self.add_intent_q_proj = nn.Linear(added_intention_dim, self.inner_dim, bias=self.added_proj_bias)
        
        # Norms
        if self.norm_added_q is not None:
            # use the same class as selfnorm_added_q by using a deepcopy
            self.norm_added_intent_q = copy.deepcopy(self.norm_added_q)
        if self.norm_added_k is not None:
            self.norm_added_intent_k = copy.deepcopy(self.norm_added_k)
        
        # out
        self.intent_to_add_out = nn.Linear(self.inner_dim, added_intention_dim, bias=self.added_proj_bias)


class IntentionAttention(Attention):
    def __init__(self, added_intention_dim, **kwargs):
        super().__init__(**kwargs)
        self.add_intent_k_proj = nn.Linear(added_intention_dim, self.inner_kv_dim, bias=self.added_proj_bias)
        self.add_intent_v_proj = nn.Linear(added_intention_dim, self.inner_kv_dim, bias=self.added_proj_bias)
        
        if self.context_pre_only is not None:
            self.add_intent_q_proj = nn.Linear(added_intention_dim, self.inner_dim, bias=self.added_proj_bias)
        
        # Norms
        if self.norm_added_q is not None:
            # use the same class as selfnorm_added_q by using a deepcopy
            self.norm_added_intent_q = copy.deepcopy(self.norm_added_q)
        if self.norm_added_k is not None:
            self.norm_added_intent_k = copy.deepcopy(self.norm_added_k)
        
        # out
        self.intent_to_add_out = nn.Linear(self.inner_dim, added_intention_dim, bias=self.added_proj_bias)



class JointAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        intention_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        
        batch_size = hidden_states.shape[0]
        inner_dim = int(hidden_states.shape[-1])
        head_dim = int(inner_dim // attn.heads)

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # `audio` projections.
        if encoder_hidden_states is not None:
            audio_query_proj = attn.add_q_proj(encoder_hidden_states)
            audio_key_proj = attn.add_k_proj(encoder_hidden_states)
            audio_value_proj = attn.add_v_proj(encoder_hidden_states)

            audio_query_proj = audio_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            audio_key_proj = audio_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            audio_value_proj = audio_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                audio_query_proj = attn.norm_added_q(audio_query_proj)
            if attn.norm_added_k is not None:
                audio_key_proj = attn.norm_added_k(audio_key_proj)

            # cat into the standard attention
            query = torch.cat([audio_query_proj, query], dim=2)
            key = torch.cat([audio_key_proj, key,], dim=2)
            value = torch.cat([audio_value_proj, value], dim=2)
        
        if intention_hidden_states is not None:
            assert attn.add_intent_q_proj is not None
            assert attn.add_intent_k_proj is not None

            intention_query_proj = attn.add_intent_q_proj(intention_hidden_states)
            intention_key_proj = attn.add_intent_k_proj(intention_hidden_states)
            intention_value_proj = attn.add_intent_v_proj(intention_hidden_states)

            if attn.norm_added_intent_q is not None:
                intention_query_proj = attn.norm_added_intent_q(intention_query_proj)
            if attn.norm_added_intent_k is not None:
                intention_key_proj = attn.norm_added_intent_k(intention_key_proj)
            
            # cat into the standard attention
            query = torch.cat([intention_query_proj, query], dim=2)
            key = torch.cat([intention_key_proj, key], dim=2)
            value = torch.cat([intention_value_proj, value], dim=2)
        
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs):
                x1, x2 = x[..., ::2], x[..., 1::2]
                rotated_x = torch.cat((-x2, x1), dim=-1)
                return (x * freqs.cos()) + (rotated_x * freqs.sin())
            
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)


        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Handle different return cases
        if encoder_hidden_states is None:
            return hidden_states
            
        # Split the hidden states
        if intention_hidden_states is not None:
            intent_len = intention_hidden_states.shape[-2]
            audio_len = encoder_hidden_states.shape[-2]
            
            intent_out, audio_out, hidden_out = (
                hidden_states[:, :intent_len],
                hidden_states[:, intent_len:intent_len+audio_len],
                hidden_states[:, intent_len+audio_len:],
            )
            
            # Apply projections
            hidden_out = attn.to_out[1](attn.to_out[0](hidden_out))
            audio_out = attn.to_add_out(audio_out)
            intent_out = attn.intent_to_add_out(intent_out)
            
            return hidden_out, audio_out, intent_out
        else:
            audio_len = encoder_hidden_states.shape[-2]
            audio_out, hidden_out = hidden_states[:, :audio_len], hidden_states[:, audio_len:]
            
            # Apply projections
            hidden_out = attn.to_out[1](attn.to_out[0](hidden_out))
            audio_out = attn.to_add_out(audio_out)
            
            return hidden_out, audio_out


class AttnProcessor:
    """Attention processor used typically in processing the attention projections."""
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        intention_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size = hidden_states.shape[0]
        inner_dim = int(hidden_states.shape[-1])
        head_dim = int(inner_dim // attn.heads)

        if encoder_hidden_states is None:
            # If no encoder_hidden_states are provided, we assume that it is self-attention.
            encoder_hidden_states = hidden_states

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs):
                x1, x2 = x[..., ::2], x[..., 1::2]  # Splitting along the last dimension
                rotated_x = torch.cat((-x2, x1), dim=-1)
                return (x * freqs.cos()) + (rotated_x * freqs.sin())

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        
        hidden_states_intention = None
        if intention_hidden_states is not None:
            key_intention = attn.add_k_proj(intention_hidden_states)
            if attn.norm_added_k is not None:
                key_intention = attn.norm_added_k(key_intention)
            value_intention = attn.add_v_proj(intention_hidden_states)

            key_intention = key_intention.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_intention = value_intention.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states_intention = F.scaled_dot_product_attention(
                query, key_intention, value_intention, attn_mask=None, dropout_p=0.0, is_causal=False)

            hidden_states_intention = hidden_states_intention.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states_intention = hidden_states_intention.to(query.dtype)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if hidden_states_intention is not None:
            # 
            hidden_states = hidden_states + hidden_states_intention
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CrossTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=AttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=AttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor = None,
        intention_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, 
            intention_hidden_states=intention_hidden_states, rotary_emb=rotary_emb
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class CrossTransformerBlockWithSeed(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # 0. Seed Cross-attention
        self.attn0 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=None,
            added_proj_bias=True,
            processor=AttnProcessor(),
        )
        

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=AttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=AttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor = None,
        intention_hidden_states: torch.Tensor = None,
        seed: torch.Tensor = None,
    ) -> torch.Tensor:
        
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)
        
        # 0. Seed Cross-attention
        seed_hidden_states = self.attn0(
            hidden_states=hidden_states, encoder_hidden_states=seed, 
            intention_hidden_states=None, rotary_emb=rotary_emb
        )
        hidden_states = hidden_states + seed_hidden_states
        
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, 
            intention_hidden_states=intention_hidden_states, rotary_emb=rotary_emb
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        cross_attn_norm: bool = False,
        activation_fn: str = "geglu",
        norm_eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        dropout: float = 0.0,
        skip: bool = False,
    ):
        super().__init__()
        
        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=AttnProcessor(),
        )
        
        # 2. Cross-Attn
        self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
        
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_heads,
            heads=num_heads,
            qk_norm=qk_norm,
            eps=1e-6,
            bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            processor=AttnProcessor(),
        )
        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=False,  ### False
            inner_dim=ffn_dim,  ### int(dim * mlp_ratio)
            bias=True,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(2 * dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0
    
    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        rotary_emb=None,
        intention_hidden_states: torch.Tensor = None,
        skip=None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([hidden_states, skip], dim=-1)
            cat = self.skip_norm(cat)
            hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states, temb)  ### checked: self.norm1 is correct
        attn_output = self.attn1(
            norm_hidden_states,
            rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            intention_hidden_states=intention_hidden_states,
            rotary_emb=rotary_emb,
        )

        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states

class DiTBlockWithSeed(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        cross_attn_norm: bool = False,
        activation_fn: str = "geglu",
        norm_eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        dropout: float = 0.0,
        skip: bool = False,
    ):
        super().__init__()
        
        # 0. Seed Cross-attention
        self.attn0 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=1e-6,
            bias=True,
            processor=AttnProcessor(),
        )
        
        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=norm_eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=AttnProcessor(),
        )
        
        # 2. Cross-Attn
        self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
        
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_heads,
            heads=num_heads,
            qk_norm=qk_norm,
            eps=1e-6,
            bias=True,
            processor=AttnProcessor(),
        )
        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=False,  ### False
            inner_dim=ffn_dim,  ### int(dim * mlp_ratio)
            bias=True,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(2 * dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0
    
    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        rotary_emb=None,
        intention_hidden_states: torch.Tensor = None,
        skip=None,
        seed: torch.Tensor = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # -1. Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([hidden_states, skip], dim=-1)
            cat = self.skip_norm(cat)
            hidden_states = self.skip_linear(cat)
        
        # 0. Seed Cross-attention
        seed_hidden_states = self.attn0(
            hidden_states=hidden_states, encoder_hidden_states=seed, 
            intention_hidden_states=None, rotary_emb=rotary_emb
        )
        hidden_states = hidden_states + seed_hidden_states

        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states, temb)  ### checked: self.norm1 is correct
        attn_output = self.attn1(
            norm_hidden_states,
            rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            intention_hidden_states=intention_hidden_states,
            rotary_emb=rotary_emb,
        )

        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states


class JointTransformerBlock(nn.Module):
    def __init__(self, 
                 dim: int, 
                 ffn_dim: int, 
                 num_heads: int, 
                 qk_norm: str = "rms_norm",
                 eps: float = 1e-6,
                 process_intention: bool = False
        ):
        super().__init__()

        
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        if process_intention:
            self.norm1_intention = AdaLayerNormZero(dim)

            # 1. Intention-attention
            self.attn = IntentionJointAttention(
                added_intention_dim=dim,
                query_dim=dim,
                heads=num_heads,
                kv_heads=num_heads,
                dim_head=dim // num_heads,
                cross_attention_dim=None,
                added_kv_proj_dim=dim,
                out_dim=dim,
                context_pre_only=False,
                bias=True,
                processor=JointAttnProcessor(),
                qk_norm=qk_norm,
                eps=eps,
            )
        else:
            # 1. Joint-attention
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_heads,
                kv_heads=num_heads,
                dim_head=dim // num_heads,
                cross_attention_dim=None,
                added_kv_proj_dim=dim,
                out_dim=dim,
                context_pre_only=False,
                bias=True,
                processor=JointAttnProcessor(),
                qk_norm=qk_norm,
                eps=eps,
            )

        # 2. Feed-forward
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = FeedForward(dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if process_intention:
            self.norm2_intention = FP32LayerNorm(dim, eps, elementwise_affine=False)
            self.ff_intention = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
            rotary_emb: torch.Tensor = None,
            intention_hidden_states: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        if hasattr(self, "norm1_intention"):
            norm_intention_hidden_states, i_gate_msa, i_shift_mlp, i_scale_mlp, i_gate_mlp = self.norm1_intention(
                intention_hidden_states, emb=temb
            )

        # Attention.
        attention_outputs = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            intention_hidden_states=norm_intention_hidden_states if hasattr(self, "norm1_intention") else None,
            rotary_emb=rotary_emb,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, intent_attn_output = attention_outputs

        
        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ffn(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        if hasattr(self, "norm1_intention"):
            # Process attention outputs for the `intention_hidden_states`.
            intent_attn_output = i_gate_msa.unsqueeze(1) * intent_attn_output
            intention_hidden_states = intention_hidden_states + intent_attn_output

            norm_intention_hidden_states = self.norm2_intention(intention_hidden_states)
            norm_intention_hidden_states = norm_intention_hidden_states * (1 + i_scale_mlp[:, None]) + i_shift_mlp[:, None]

            intent_ff_output = self.ff_intention(norm_intention_hidden_states)
            intention_hidden_states = intention_hidden_states + i_gate_mlp.unsqueeze(1) * intent_ff_output

            return hidden_states, encoder_hidden_states, intention_hidden_states

        return encoder_hidden_states, hidden_states