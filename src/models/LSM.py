import time
import inspect
import logging
from typing import Optional

import scipy.stats as stats
import tqdm
import numpy as np
from omegaconf import DictConfig
from typing import Dict
import math
import torch
import torch.distributions as dist

import torch
import torch.nn.functional as F
from models.config import instantiate_from_config
from models.utils.utils import count_parameters, extract_into_tensor, sum_flat

logger = logging.getLogger(__name__)

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)

def sample_t(exponential_pdf, num_samples, a=2):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    return t

def sample_beta_distribution(num_samples, alpha=2, beta=0.8, t_min=1e-5, t_max=1-1e-5):
    """
    Samples from a Beta distribution with the specified parameters.
    
    Args:
        num_samples (int): Number of samples to generate.
        alpha (float): Alpha parameter of the Beta distribution (shape1).
        beta (float): Beta parameter of the Beta distribution (shape2).
        t_min (float): Minimum value for scaling the samples (default is near 0).
        t_max (float): Maximum value for scaling the samples (default is near 1).
        
    Returns:
        torch.Tensor: Tensor of sampled values.
    """
    # Define the Beta distribution
    beta_dist = dist.Beta(alpha, beta)
    
    # Sample values from the Beta distribution
    samples = beta_dist.sample((num_samples,))
    
    # Scale the samples to the range [t_min, t_max]
    scaled_samples = samples * (t_max - t_min) + t_min
    
    return scaled_samples

def sample_t_fast(num_samples, a=2, t_min=1e-5, t_max=1-1e-5):
    # Direct inverse sampling for exponential distribution
    C = a / (np.exp(a) - 1)
    
    # Generate uniform samples
    u = torch.rand(num_samples * 2)
    
    # Inverse transform sampling formula for the exponential PDF
    # F^(-1)(u) = (1/a) * ln(1 + u*(exp(a) - 1))
    t = (1/a) * torch.log(1 + u * (np.exp(a) - 1))
    
    # Combine t and 1-t
    t = torch.cat([t, 1 - t])
    
    # Random permutation and slice
    t = t[torch.randperm(t.shape[0])][:num_samples]
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def sample_cosmap(num_samples, t_min=1e-5, t_max=1-1e-5, device='cpu'):
    """
    CosMap sampling.
    Args:
        num_samples: Number of samples to generate
        t_min, t_max: Range limits to avoid numerical issues
    """
    # Generate uniform samples
    u = torch.rand(num_samples, device=device)
    
    # Apply the cosine mapping
    pi_half = torch.pi / 2
    t = 1 - 1 / (torch.tan(pi_half * u) + 1)
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def reshape_coefs(t):
    return t.reshape((t.shape[0], 1, 1, 1))

class GestureLSM(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # Initialize model components
        self.modality_encoder = instantiate_from_config(cfg.model.modality_encoder)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        # Model hyperparameters
        self.do_classifier_free_guidance = cfg.model.do_classifier_free_guidance
        self.guidance_scale = cfg.model.guidance_scale
        self.num_inference_steps = cfg.model.n_steps
        self.exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')
        self.input_dim = self.denoiser.input_dim

        # Loss functions
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')
        
        self.num_joints = 3 if not self.cfg.model.use_exp else 4
        
        self.seq_len = self.cfg.model.denoiser.params.seq_len
        self.raw_audio = self.cfg.model.raw_audio
    def summarize_parameters(self) -> None:
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')
        logger.info(f'Encoder: {count_parameters(self.modality_encoder)}M')
    
    def forward(self, condition_dict: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """Forward pass for inference.
        
        Args:
            condition_dict: Dictionary containing input conditions including audio, word tokens,
                          and other features
        
        Returns:
            Dictionary containing generated latents
        """
        # Extract input features
        in_audio = condition_dict['y']['audio_tensor']
        cached_audio_low = condition_dict['y']['audio_low']
        cached_audio_high = condition_dict['y']['audio_high']
        instance_ids = condition_dict['y']['id']
        seed_vectors = condition_dict['y']['seed']
        style_features = condition_dict['y']['style_feature']
        intention_embeddings = condition_dict['y']['intention_embeddings']
        intention_mask = condition_dict['y']['intention_mask']
        audio_onset = None
        if 'audio_onset' in condition_dict['y']:
            audio_onset = condition_dict['y']['audio_onset']
        if 'word' in condition_dict['y']:
            word = condition_dict['y']['word']
        return_dict = {}
        return_dict['seed'] = seed_vectors
        
        # Encode input modalities
        if audio_onset is not None and self.raw_audio:
            audio_features = self.modality_encoder(in_audio, cached_audio_low, cached_audio_high, intention_embeddings, intention_mask, audio_onset, word)
        else:
            audio_features = self.modality_encoder(in_audio, cached_audio_low, cached_audio_high, intention_embeddings, intention_mask, audio_onset, word)
        at_feat = audio_features['audio_low']
        intent_feat = audio_features['audio_high']
        return_dict['at_feat'] = at_feat
        return_dict['intent_feat'] = intent_feat

        # Initialize generation
        batch_size = at_feat.shape[0]
        latent_shape = (batch_size, self.input_dim, 1, self.seq_len)

        
        
        # Sampling parameters
        x_t = torch.randn(latent_shape, device=at_feat.device)

        return_dict['init_noise'] = x_t
        
        epsilon = 1e-8
        delta_t = torch.tensor(1 / self.num_inference_steps).to(at_feat.device)
        timesteps = torch.linspace(epsilon, 1 - epsilon, self.num_inference_steps + 1).to(at_feat.device)
        
        # Generation loop
        for step in range(1, len(timesteps)):
            current_t = timesteps[step - 1].unsqueeze(0)
            current_delta = delta_t.unsqueeze(0)
            
            with torch.no_grad():
                speed = self.denoiser.forward_with_cfg(
                    x=x_t,
                    timesteps=current_t,
                    seed=seed_vectors,
                    at_feat=at_feat,
                    intent_feat=intent_feat if intent_feat is not None else None,
                    cond_time=current_delta,
                    instance_ids=instance_ids,
                    style_features=style_features,
                    guidance_scale=self.guidance_scale
                )
               
            x_t = x_t + (timesteps[step] - timesteps[step - 1]) * speed
        return_dict['latents'] = x_t
        return return_dict
    
    def forward_calculate_loss(self, condition_dict: Dict[str, Dict], latents: torch.Tensor, save_path: str, iter: int) -> Dict[str, torch.Tensor]:
        """Compute losses for the forward pass.
        
        Args:
            condition_dict: Dictionary containing input conditions
            latents: Target latent vectors
        
        Returns:
            Dictionary containing individual and total losses
        """
        # Extract input features
        audio = condition_dict['y']['audio']
        raw_audio = condition_dict['y']['wavlm']
        word_tokens = condition_dict['y']['word']
        instance_ids = condition_dict['y']['id']
        seed_vectors = condition_dict['y']['seed']
        style_features = condition_dict['y']['style_feature']
        intention_embeddings = condition_dict['y']['intention_embeddings']
        intention_mask = condition_dict['y']['intention_mask']
    
        # Encode input modalities
        audio_features = self.modality_encoder(audio, word_tokens, raw_audio, intention_embeddings, intention_mask)

        # Initialize noise
        x0_noise = torch.randn_like(latents)

        # Sample timesteps and deltas
        deltas = 1 / torch.tensor([2 ** i for i in range(1, 8)]).to(latents.device)
        delta_probs = torch.ones((deltas.shape[0],)).to(latents.device) / deltas.shape[0]

        batch_size = latents.shape[0]
        flow_batch_size = int(batch_size * 3/4)

        # Sample random coefficients
        epsilon = 1e-8

        timesteps = torch.linspace(epsilon, 1 - epsilon, 50 + 1).to(audio_features.device)

        losses = {}
        for step in range(1, len(timesteps)):
            t = timesteps[step - 1].unsqueeze(0).repeat((batch_size,))

            # t = sample_t_fast(batch_size).to(latents.device)
            d = deltas[delta_probs.multinomial(batch_size, replacement=True)]
            d[:flow_batch_size] = 0

            # Prepare inputs
            t_coef = reshape_coefs(t)
            x_t = t_coef * latents + (1 - t_coef) * x0_noise
            t = t_coef.flatten()
            
            # Flow matching loss
            flow_pred = self.denoiser(
                x=x_t[:flow_batch_size],
                timesteps=t[:flow_batch_size],
                seed=seed_vectors[:flow_batch_size],
                at_feat=audio_features[:flow_batch_size],
                cond_time=d[:flow_batch_size],
                instance_ids=instance_ids[:flow_batch_size],
                style_features=style_features[:flow_batch_size],
            )
            
            flow_target = latents[:flow_batch_size] - x0_noise[:flow_batch_size]
            
            flow_loss = F.mse_loss(flow_target, flow_pred, reduction='none').mean(dim=(1, 2, 3))
            
            losses[t[0].item()] = flow_loss.tolist()


        #plot this loss
        # save this loss into a csv file
        with open(save_path + f'loss_{iter}.csv', 'w') as f:
            for key, loss_vals in losses.items():
                loss_vals_str = "\t".join(map(str, loss_vals))
                f.write(f"{key}\t{loss_vals_str}\n")

        return losses
    
    def train_forward(self, condition_dict: Dict[str, Dict], 
                              latents: torch.Tensor, train_consistency=False) -> Dict[str, torch.Tensor]:
        """Compute training losses for both flow matching and consistency.
        
        Args:
            condition_dict: Dictionary containing training conditions
            latents: Target latent vectors
            
        Returns:
            Dictionary containing individual and total losses
        """

        # Extract input features
        in_audio = condition_dict['y']['audio_tensor']
        cached_audio_low = condition_dict['y']['audio_low']
        cached_audio_high = condition_dict['y']['audio_high']
        audio_onset = None
        if 'audio_onset' in condition_dict['y']:
            audio_onset = condition_dict['y']['audio_onset']
        if 'word' in condition_dict['y']:
            word = condition_dict['y']['word']
        else:
            word = None
        
        instance_ids = condition_dict['y']['id']
        seed_vectors = condition_dict['y']['seed']
        style_features = condition_dict['y']['style_feature']
        intention_embeddings = condition_dict['y']['intention_embeddings']
        intention_mask = condition_dict['y']['intention_mask']
        
        # Encode input modalities
        if audio_onset is not None and self.raw_audio:
            audio_features = self.modality_encoder(in_audio, cached_audio_low, cached_audio_high, intention_embeddings, intention_mask, audio_onset, word)
        else:
            audio_features = self.modality_encoder(in_audio, cached_audio_low, cached_audio_high, intention_embeddings, intention_mask, audio_onset, word)
        at_feat = audio_features['audio_low']
        intent_feat = audio_features['audio_high']

        # Initialize noise
        x0_noise = torch.randn_like(latents)

        # Sample timesteps and deltas
        deltas = 1 / torch.tensor([2 ** i for i in range(1, 8)]).to(latents.device)
        delta_probs = torch.ones((deltas.shape[0],)).to(latents.device) / deltas.shape[0]

        batch_size = latents.shape[0]
        flow_batch_size = int(batch_size * 3/4)

        # Sample random coefficients
        # t = sample_beta_distribution(batch_size, alpha=2, beta=1.2).to(latents.device)
        # t = sample_beta_distribution(batch_size, alpha=2, beta=0.8).to(latents.device)
        t = torch.sigmoid(torch.randn(batch_size, device=latents.device))
        d = deltas[delta_probs.multinomial(batch_size, replacement=True)]
        d[:flow_batch_size] = 0

        # Prepare inputs
        t_coef = reshape_coefs(t)
        x_t = t_coef * latents + (1 - t_coef) * x0_noise
        t = t_coef.flatten()
        
        # Flow matching loss
        flow_pred = self.denoiser(
            x=x_t[:flow_batch_size],
            timesteps=t[:flow_batch_size],
            seed=seed_vectors[:flow_batch_size],
            at_feat=at_feat[:flow_batch_size],
            intent_feat=intent_feat[:flow_batch_size] if intent_feat is not None else None,
            cond_time=d[:flow_batch_size],
            instance_ids=instance_ids[:flow_batch_size],
            style_features=style_features[:flow_batch_size] if style_features is not None else None,
        )
        
        flow_target = latents[:flow_batch_size] - x0_noise[:flow_batch_size]
        
        losses = {}
        flow_loss = (F.mse_loss(flow_target, flow_pred) / t).mean()
        losses['flow_loss'] = flow_loss

        # Consistency loss computation
        # Jan 11, perform cfg at the same time, 50% true and 50% false
        force_cfg = np.random.choice([True, False], size=batch_size-flow_batch_size, p=[0.5, 0.5])
        with torch.no_grad():
            speed_t = self.denoiser(
                x=x_t[flow_batch_size:],
                timesteps=t[flow_batch_size:],
                seed=seed_vectors[flow_batch_size:],
                at_feat=at_feat[flow_batch_size:],
                intent_feat=intent_feat[flow_batch_size:] if intent_feat is not None else None,
                cond_time=d[flow_batch_size:],
                instance_ids=instance_ids[flow_batch_size:],
                style_features=style_features[flow_batch_size:] if style_features is not None else None,
                force_cfg=force_cfg,
            )
            
            d_coef = reshape_coefs(d)
            x_td = x_t[flow_batch_size:] + d_coef[flow_batch_size:] * speed_t
            d = d_coef.flatten()

            speed_td = self.denoiser(
                x=x_td,
                timesteps=t[flow_batch_size:] + d[flow_batch_size:],
                seed=seed_vectors[flow_batch_size:],
                at_feat=at_feat[flow_batch_size:],
                intent_feat=intent_feat[flow_batch_size:] if intent_feat is not None else None ,
                cond_time=d[flow_batch_size:],
                instance_ids=instance_ids[flow_batch_size:],
                style_features=style_features[flow_batch_size:] if style_features is not None else None,
                force_cfg=force_cfg,
            )
            
            speed_target = (speed_t + speed_td) / 2
        
        speed_pred = self.denoiser(
            x=x_t[flow_batch_size:],
            timesteps=t[flow_batch_size:],
            seed=seed_vectors[flow_batch_size:],
            at_feat=at_feat[flow_batch_size:],
            intent_feat=intent_feat[flow_batch_size:] if intent_feat is not None else None,
            cond_time=2 * d[flow_batch_size:],
            force_cfg=force_cfg,
            instance_ids=instance_ids[flow_batch_size:],
            style_features=style_features[flow_batch_size:] if style_features is not None else None,
        )
        
        consistency_loss = F.mse_loss(speed_pred, speed_target, reduction="mean")
        losses['consistency_loss'] = consistency_loss

        losses['loss'] = sum(losses.values())
        return losses
    

    def train_reflow(self, latents, audio_features, x0_noise, seed_vectors) -> Dict[str, torch.Tensor]:
        """Compute training losses for both flow matching and consistency.
        
        Args:
            condition_dict: Dictionary containing training conditions
            latents: Target latent vectors
            
        Returns:
            Dictionary containing individual and total losses
        """

        # Sample timesteps and deltas
        deltas = 1 / torch.tensor([2 ** i for i in range(1, 8)]).to(latents.device)
        delta_probs = torch.ones((deltas.shape[0],)).to(latents.device) / deltas.shape[0]

        batch_size = latents.shape[0]
        flow_batch_size = int(batch_size * 3/4)

        # Sample random coefficients
        t = sample_beta_distribution(batch_size, alpha=2, beta=1.2).to(latents.device)
        # t = sample_beta_distribution(batch_size, alpha=2, beta=0.8).to(latents.device)
        d = deltas[delta_probs.multinomial(batch_size, replacement=True)]
        d[:flow_batch_size] = 0

        # Prepare inputs
        t_coef = reshape_coefs(t)
        x_t = t_coef * latents + (1 - t_coef) * x0_noise
        t = t_coef.flatten()
        
        # Flow matching loss
        flow_pred = self.denoiser(
            x=x_t[:flow_batch_size],
            timesteps=t[:flow_batch_size],
            seed=seed_vectors[:flow_batch_size],
            at_feat=at_feat[:flow_batch_size],
            intent_feat=intent_feat[:flow_batch_size] if intent_feat is not None else None,
            cond_time=d[:flow_batch_size],
        )
        
        flow_target = latents[:flow_batch_size] - x0_noise[:flow_batch_size]
        
        losses = {}
        flow_loss = (F.mse_loss(flow_target, flow_pred) / t).mean()
        losses['flow_loss'] = flow_loss

        # Consistency loss computation
        # Jan 11, perform cfg at the same time, 50% true and 50% false
        force_cfg = np.random.choice([True, False], size=batch_size-flow_batch_size, p=[0.8, 0.2])
        with torch.no_grad():
            speed_t = self.denoiser(
                x=x_t[flow_batch_size:],
                timesteps=t[flow_batch_size:],
                seed=seed_vectors[flow_batch_size:],
                at_feat=at_feat[flow_batch_size:],
                intent_feat=intent_feat[flow_batch_size:] if intent_feat is not None else None,
                cond_time=d[flow_batch_size:],
                force_cfg=force_cfg,
            )
            
            d_coef = reshape_coefs(d)
            x_td = x_t[flow_batch_size:] + d_coef[flow_batch_size:] * speed_t
            d = d_coef.flatten()

            speed_td = self.denoiser(
                x=x_td,
                timesteps=t[flow_batch_size:] + d[flow_batch_size:],
                seed=seed_vectors[flow_batch_size:],
                at_feat=audio_features[flow_batch_size:],
                cond_time=d[flow_batch_size:],
                force_cfg=force_cfg,
            )
            
            speed_target = (speed_t + speed_td) / 2
        
        speed_pred = self.denoiser(
            x=x_t[flow_batch_size:],
            timesteps=t[flow_batch_size:],
            seed=seed_vectors[flow_batch_size:],
            at_feat=audio_features[flow_batch_size:],
            cond_time=2 * d[flow_batch_size:],
            force_cfg=force_cfg,
        )
        
        consistency_loss = F.mse_loss(speed_pred, speed_target, reduction="mean")
        losses['consistency_loss'] = consistency_loss

        losses['loss'] = sum(losses.values())
        return losses


    
    def huber_loss(self, a, b, reduction='mean'):
        data_dim = a.shape[1] * a.shape[2] * a.shape[3]
        huber_c = 0.00054 * data_dim
        loss = torch.sum((a - b) ** 2, dim=(1, 2, 3))
        loss = torch.sqrt(loss + huber_c**2) - huber_c
        loss = loss / data_dim
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss