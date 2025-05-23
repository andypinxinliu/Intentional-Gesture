import random
from math import ceil
from functools import partial
from itertools import zip_longest
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from models.vq.dual_quantizer import DualVectorQuantizer

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

class DualResidualVQ(nn.Module):
    """ Modified RVQ implementation using DualVectorQuantizer """
    def __init__(
        self,
        num_quantizers,
        nb_code,
        split,
        l2_norm=True,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        show_usage=True,
        entropy_loss_ratio=0.1,
        beta=0.25,
        projection=False
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        
        # Create the quantizer layers
        if shared_codebook:
            layer = DualVectorQuantizer(
                nb_code=nb_code,
                split=split,
                l2_norm=l2_norm,
                show_usage=show_usage,
                entropy_loss_ratio=entropy_loss_ratio,
                beta=beta,
                projection=projection
            )
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([
                DualVectorQuantizer(
                    nb_code=nb_code,
                    split=split,
                    l2_norm=l2_norm,
                    show_usage=show_usage,
                    entropy_loss_ratio=entropy_loss_ratio,
                    beta=beta,
                    projection=projection
                ) for _ in range(num_quantizers)
            ])

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

    def get_codebook_entry(self, indices, shape=None):
        """
        Reconstruct input from indices
        Args:
            indices: tensor of shape [batch, *spatial, num_quantizers]
        """
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # Handle quantize dropout case
        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        quantized_out = 0.
        for quantizer_index, (layer, idx) in enumerate(zip(self.layers, indices.transpose(-1, 0))):
            # Skip dropped out quantizers
            if (idx == -1).all():
                continue
                
            quantized = layer.get_codebook_entry(idx, shape)
            quantized_out = quantized_out + quantized

        return quantized_out

    def forward(self, x, return_indices_only=False, force_dropout_index=-1):
        num_quant, quant_dropout_prob, device = self.num_quantizers, self.quantize_dropout_prob, x.device
        
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_auxiliary = []

        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant)
            null_indices_shape = (x.shape[0], *x.shape[2:])  # [b, h, w] for images
            null_indices = torch.full(null_indices_shape, -1, device=device, dtype=torch.long)

        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = (x.shape[0], *x.shape[2:])
            null_indices = torch.full(null_indices_shape, -1, device=device, dtype=torch.long)

        # Process through quantizer layers
        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue

            quantized, losses, auxiliary = layer(residual)
            
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            # Unpack results
            _, _, _, _, vqkd_d_norm, vqgan_d_norm = losses
            perplexity, min_encodings, indices = auxiliary
            
            all_indices.append(indices)
            all_losses.append(losses)
            all_auxiliary.append(auxiliary)

        # Stack indices and combine losses
        all_indices = torch.stack(all_indices, dim=-1)  # [b, h, w, num_quantizers]
        
        if return_indices_only:
            return all_indices

        # Combine losses from all layers
        vq_loss = sum(loss[0] for loss in all_losses if loss[0] is not None) / len(all_losses)
        commit_loss = sum(loss[1] for loss in all_losses if loss[1] is not None) / len(all_losses)
        entropy_loss = sum(loss[2] for loss in all_losses if loss[2] is not None) / len(all_losses)
        
        # Average codebook usage across layers
        codebook_usage = sum(loss[3] for loss in all_losses if loss[3] is not None) / len(all_losses)
        
        # Average distance norms
        vqkd_d_norm = sum(loss[4] for loss in all_losses) / len(all_losses)
        vqgan_d_norm = sum(loss[5] for loss in all_losses) / len(all_losses)

        combined_losses = (vq_loss, commit_loss, entropy_loss, codebook_usage, vqkd_d_norm, vqgan_d_norm)
        
        return quantized_out, all_indices, combined_losses