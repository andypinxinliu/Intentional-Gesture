import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding='same')
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding='same')
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x.transpose(1,2)).transpose(1,2)
        x = self.norm1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x.transpose(1,2)).transpose(1,2)
        x = self.norm2(x)
        return F.leaky_relu(x + residual, 0.2)

class MotionDiscriminator(nn.Module):
    def __init__(self, cfg=None):
        super(MotionDiscriminator, self).__init__()
        input_dim = cfg.get('input_channels', 825)
        hidden_dim = cfg.get('hidden_dim', 512)
        num_layers = cfg.get('num_layers', 4)
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Temporal convolution blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, kernel_size=3, dilation=2**i) 
            for i in range(num_layers)
        ])
        
        # Global temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Final classification
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels)
        Returns:
            logits: Classification logits
            features: Feature maps for feature matching loss
        """
        features = []
        
        # Initial projection
        x = self.input_proj(x)
        features.append(x)
        
        # Temporal processing
        for res_block in self.res_blocks:
            x = res_block(x)
            features.append(x)
        
        # Attention-based temporal pooling
        attention_weights = self.attention(x)
        x = torch.sum(x * attention_weights, dim=1)
        
        # Final classification
        logits = self.output(x)
        
        return logits, features

class MultiScaleMotionDiscriminator(nn.Module):
    def __init__(self, cfg=None):
        super(MultiScaleMotionDiscriminator, self).__init__()
        
        self.discriminators = nn.ModuleList([
            MotionDiscriminator(cfg) for _ in range(3)
        ])
        
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels)
        Returns:
            list of discriminator outputs at different scales
        """
        results = []
        
        for i, D in enumerate(self.discriminators):
            if i != 0:
                # Downsample the sequence length
                x = x.transpose(1, 2)
                x = self.downsample(x)
                x = x.transpose(1, 2)
            logits, _ = D(x)
            results.append(logits)
            
        return torch.cat(results, dim=1)