import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.nn.functional import scaled_dot_product_attention

from models.vq.m_quantizer import VectorQuantizerM
from models.layers.motion_encoder import WrapedMotionEncoder, WrapedMotionDecoder
from models.layers.transformer import Mlp


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = scaled_dot_product_attention(q, k, v)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = Attention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=out_dim,
            hidden_features=hidden_dim,
            out_features=out_dim,
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 1. build encoder
        self.encoder = WrapedMotionEncoder(
            args.model,
        )
        self.encoder.set_grad_checkpointing(args.grad_ckpt)

        # 2. build conv before quant
        if args.quant_proj == 'linear':
            self.quant_proj = nn.Linear(self.encoder.embed_dim, args.vocab_width)
        elif args.quant_proj == 'attn':
            self.quant_proj = AttnProjection(self.encoder.embed_dim, args.vocab_width, args.num_codebooks)
        else:
            raise NotImplementedError

        # 3. build quant
        self.quantize = VectorQuantizerM(
            vocab_size=args.vocab_size,
            vocab_width=args.vocab_width,
            beta=args.vq_beta,
            use_entropy_loss=args.le > 0,
            entropy_temp=args.e_temp,
            num_codebooks=args.num_codebooks,
        )

        # 4. build conv after quant
        if args.quant_proj == 'linear':
            self.post_quant_proj = nn.Linear(args.vocab_width, self.encoder.embed_dim)
        elif args.quant_proj == 'attn':
            self.post_quant_proj = AttnProjection(args.vocab_width, self.encoder.embed_dim, args.num_codebooks)
        else:
            raise NotImplementedError

        # 5. build decoder
        self.decoder = WrapedMotionDecoder(
            args.model,
        )

        self.maybe_record_function = nullcontext

    def forward(self, motion):
        features = self.encoder(motion).float()
        with torch.cuda.amp.autocast(enabled=False):
            features = self.quant_proj(features)
            quant_out = self.quantize(features)
            features, vq_loss, entropy_loss, usages = quant_out
            features = self.post_quant_proj(features)
        rec_motion = self.decoder(features).float()
        return rec_motion, vq_loss, entropy_loss, usages

    def gesture_to_idx(self, motion):
        features = self.encoder(motion).float()
        features = self.quant_proj(features)
        return self.quantize.f_to_idx(features)

    def idx_to_motion(self, indices):
        features = self.quantize.idx_to_f(indices)
        features = self.post_quant_proj(features)
        motion = self.decoder(features).clamp_(-1, 1)
        return motion

    def motion_to_reconstructed_motion(self, motion) -> torch.Tensor:
        features = self.encoder(motion).float()
        with torch.cuda.amp.autocast(enabled=False):
            features = self.quant_proj(features)
            quant_out = self.quantize(features)
            features, _, _, _ = quant_out
            features = self.post_quant_proj(features)
        rec_motion = self.decoder(features).float().clamp_(-1, 1)
        return rec_motion


if __name__ == '__main__':
    import torch
    import argparse
    import sys
    import os
    import numpy as np
    from pathlib import Path
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Test VQVAE model")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size for testing")
        parser.add_argument("--seq_len", type=int, default=120, help="Sequence length for motion")
        parser.add_argument("--motion_dim", type=int, default=135, help="Dimension of motion features")
        parser.add_argument("--motion_f", type=int, default=512, help="Motion feature dimension")
        parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size for transformer")
        parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
        parser.add_argument("--vocab_size", type=int, default=1024, help="Vocabulary size for VQ")
        parser.add_argument("--vocab_width", type=int, default=512, help="Width of vocabulary embeddings")
        parser.add_argument("--num_codebooks", type=int, default=1, help="Number of codebooks")
        parser.add_argument("--quant_proj", type=str, default="linear", help="Quantization projection type")
        return parser.parse_args()
    
    def create_dummy_model_args(args):
        class ModelArgs:
            def __init__(self):
                self.motion_dim = args.motion_dim
                self.motion_f = args.motion_f
                self.hidden_size = args.hidden_size
                self.n_layer = args.n_layer
                self.vae_layer = 3
                self.vae_length = args.motion_f
                self.vae_test_dim = args.motion_dim
        
        class Args:
            def __init__(self):
                self.vocab_size = args.vocab_size
                self.vocab_width = args.vocab_width
                self.num_codebooks = args.num_codebooks
                self.quant_proj = args.quant_proj
                self.model = ModelArgs()
        
        return Args()
    
    def test_vqvae():
        args = parse_args()
        
        # Create dummy model arguments
        model_args = create_dummy_model_args(args)
        
        # Create random motion data
        batch_size = args.batch_size
        seq_len = args.seq_len
        motion_dim = args.motion_dim
        
        # Create random motion tensor
        motion = torch.randn(batch_size, seq_len, motion_dim)
        
        # Initialize model
        print("Initializing VQVAE model...")
        model = VQVAE(model_args)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        motion = motion.to(device)
        
        print(f"Model created successfully. Testing on {device}...")
        
        # Forward pass
        with torch.no_grad():
            rec_motion, vq_loss, entropy_loss, usages = model(motion)
        
        # Print results
        print(f"Input motion shape: {motion.shape}")
        print(f"Reconstructed motion shape: {rec_motion['motion'].shape}")
        print(f"VQ loss: {vq_loss.item()}")
        print(f"Entropy loss: {entropy_loss.item()}")
        print(f"Codebook usage: {usages}")
        
        # Test gesture_to_idx and idx_to_motion
        with torch.no_grad():
            indices = model.gesture_to_idx(motion)
            reconstructed = model.idx_to_motion(indices)
        
        print(f"Indices shape: {indices.shape}")
        print(f"Reconstructed from indices shape: {reconstructed.shape}")
        
        # Test motion_to_reconstructed_motion
        with torch.no_grad():
            direct_reconstruction = model.motion_to_reconstructed_motion(motion)
        
        print(f"Direct reconstruction shape: {direct_reconstruction.shape}")
        
        # Calculate reconstruction error
        mse = torch.mean((motion - rec_motion['motion']) ** 2)
        print(f"MSE between input and reconstruction: {mse.item()}")
        
        return model, motion, rec_motion

    test_vqvae()

