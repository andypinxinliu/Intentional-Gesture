import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
import torch.distributed as dist

class DualVectorQuantizer(nn.Module):
    def __init__(self, nb_code, split, l2_norm, show_usage, entropy_loss_ratio, beta, projection=False):
        super(self).__init__()
        self.nb_code = nb_code
        self.split = split

        self.sem_dim = split[0]
        self.vq_dim = split[1]
        self.e_dim = self.sem_dim + self.vq_dim

        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.entropy_loss_ratio = entropy_loss_ratio
        self.beta = beta

        self.projection = projection
        if self.projection:
            self.embedding_sem_proj = nn.Linear(split[0], split[0])
            self.embedding_vq_proj = nn.Linear(split[1], split[1])
        

        self.embedding_vqkd = nn.Embedding(self.nb_code, self.split[0])
        self.embedding_vqkd.weight.data.uniform_(-1.0 / self.nb_code, 1.0 / self.nb_code)
        if self.l2_norm:
            self.embedding_vqkd.weight.data = F.normalize(self.embedding_vqkd.weight.data, p=2, dim=-1)

        self.embedding_vqgan = nn.Embedding(self.nb_code, self.split[1])
        self.embedding_vqgan.weight.data.uniform_(-1.0 / self.nb_code, 1.0 / self.nb_code)
        if self.l2_norm:
            self.embedding_vqgan.weight.data = F.normalize(self.embedding_vqgan.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(self.nb_code)))

    def forward(self, z):
        if z.dim() == 3:
            z = torch.einsum('b d l -> b l d', z).contiguous()
        elif z.dim() == 4:
            z = torch.einsum('b c h w -> b h w c', z).contiguous()
        else:
            raise ValueError("Input shape not supported")
        
        z_flattened = z.view(-1, self.e_dim)

        z_vqkd, z_vqgan = torch.split(z, split_size_or_sections=self.split, dim=-1)
        z_flattened_vqkd, z_flattened_vqgan = torch.split(z_flattened, split_size_or_sections=self.split, dim=-1)

        if self.l2_norm:
            z_flattened_vqkd = F.normalize(z_flattened_vqkd, p=2, dim=-1)
            z_flattened_vqgan = F.normalize(z_flattened_vqgan, p=2, dim=-1)
            z_flattened = torch.cat([z_flattened_vqkd, z_flattened_vqgan], dim=-1)
        
        if self.l2_norm:
            z_vqkd = F.normalize(z_vqkd, p=2, dim=-1)
            embedding_vqkd = F.normalize(self.embedding_vqkd.weight, p=2, dim=-1)

            z_vqgan = F.normalize(z_vqgan, p=2, dim=-1)
            embedding_vqgan = F.normalize(self.embedding_vqgan.weight, p=2, dim=-1)
            z = torch.cat([z_vqkd, z_vqgan], dim=-1)

        else:
            embedding_vqkd = self.embedding_vqkd.weight
            embedding_vqgan = self.embedding_vqgan.weight
        
        if self.projection:
            embedding_vqgan = self.embedding_vq_proj(embedding_vqgan)
            embedding_vqkd = self.embedding_sem_proj(embedding_vqkd)


        d_vqkd = torch.sum(z_flattened_vqkd ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding_vqkd**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened_vqkd, torch.einsum('n d -> d n', embedding_vqkd))
        d_vqgan = torch.sum(z_flattened_vqgan ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding_vqgan**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened_vqgan, torch.einsum('n d -> d n', embedding_vqgan))
        
        vqkd_d_norm = torch.mean(torch.sum(d_vqkd**2, dim=-1))
        vqgan_d_norm = torch.mean(torch.sum(d_vqgan**2, dim=-1))
        
        ### shared mapping ###
        d = d_vqkd + 1.0 * d_vqgan
        min_encoding_indices = torch.argmin(d, dim=1)
        ### shared mapping ###

        aggregate_usage = False
        if aggregate_usage:
            with torch.no_grad():
                min_encoding_indices_all = [torch.zeros_like(min_encoding_indices) for _ in range(torch.distributed.get_world_size())]
                dist.all_gather(min_encoding_indices_all, min_encoding_indices)
                min_encoding_indices_all = torch.cat(min_encoding_indices_all, dim=0)

        all_embedding = torch.cat([embedding_vqkd, embedding_vqgan], dim=-1)

        z_q = all_embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            if aggregate_usage:
                cur_len = min_encoding_indices_all.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices_all
                codebook_usage = len(torch.unique(self.codebook_used)) / self.nb_code
            else:
                cur_len = min_encoding_indices.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices
                codebook_usage = len(torch.unique(self.codebook_used)) / self.nb_code
            

        # compute loss for embedding
        if self.training:
            
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage, vqkd_d_norm, vqgan_d_norm), (perplexity, min_encodings, min_encoding_indices)
    

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding_vqkd = F.normalize(self.embedding_vqkd.weight, p=2, dim=-1)
            embedding_vqgan = F.normalize(self.embedding_vqgan.weight, p=2, dim=-1)
        else:
            embedding_vqkd = self.embedding_vqkd.weight
            embedding_vqgan = self.embedding_vqgan.weight
        if self.projection:
            embedding_vqgan = self.embedding_vq_proj(embedding_vqgan)
            embedding_vqkd = self.embedding_sem_proj(embedding_vqkd)
            
        embedding = torch.cat([embedding_vqkd, embedding_vqgan], dim=-1)
        
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q
    
    def get_codebook_entry_outside(self, indices, outside_embedding, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding_vqkd = F.normalize(outside_embedding[0].weight, p=2, dim=-1)
            embedding_vqgan = F.normalize(outside_embedding[1].weight, p=2, dim=-1)
        else:
            embedding_vqkd = outside_embedding[0].weight
            embedding_vqgan = outside_embedding[1].weight
        if self.projection:
            embedding_vqgan = self.embedding_vq_proj(embedding_vqgan)
            embedding_vqkd = self.embedding_sem_proj(embedding_vqkd)

            embedding = torch.cat([embedding_vqkd, embedding_vqgan], dim=-1)
        
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
