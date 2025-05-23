import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import copy
from typing import Optional
from models.layers.layer import ResBlock, Resnet1D, CustomTransformerEncoderLayer, Stem
from models.layers.utils import init_weight
from ..utils.skeleton import SkeletonResidual, SkeletonConv, SkeletonPool, find_neighbor
from models.layers.utils import PositionalEncoding
import os
from torch.utils.checkpoint import checkpoint_sequential
from functools import partial

class Encoder1D(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.model:
            x = layer(x)
        x = x.permute(0, 2, 1)
        return x


class Decoder1D(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.model(x)
        return x.permute(0, 2, 1)
    

class VQEncoderV3(nn.Module):
    def __init__(self, args):
        super(VQEncoderV3, self).__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

class VQEncoderV6(nn.Module):
    def __init__(self, args):
        super(VQEncoderV6, self).__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]
        for i in range(n_down-1):
            channels.append(args.vae_length)
        
        input_size = args.vae_test_dim
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs


class VQDecoderV3(nn.Module):
    def __init__(self, args):
        super(VQDecoderV3, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.vae_test_dim)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs

class VQDecoderV6(nn.Module):
    def __init__(self, args):
        super(VQDecoderV6, self).__init__()
        n_up = args.vae_layer
        channels = []
        for i in range(n_up-1):
            channels.append(args.vae_length)
        channels.append(args.vae_length)
        channels.append(args.vae_test_dim)
        input_size = args.vae_length
        n_resblk = 2
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs


class LocalEncoder(nn.Module):
    def __init__(self, args, topology):
        super(LocalEncoder, self).__init__()
        args.channel_base = 6
        args.activation = "tanh"
        args.use_residual_blocks=True
        args.z_dim=1024
        args.temporal_scale=8
        args.kernel_size=4
        args.num_layers=args.vae_layer
        args.skeleton_dist=2
        args.extra_conv=0
        # check how to reflect in 1d
        args.padding_mode="constant"
        args.skeleton_pool="mean"
        args.upsampling="linear"


        self.topologies = [topology]
        self.channel_base = [args.channel_base]

        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        # self.convs = []

        kernel_size = args.kernel_size
        kernel_even = False if kernel_size % 2 else True
        padding = (kernel_size - 1) // 2
        bias = True
        self.grow = args.vae_grow
        for i in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1]*self.grow[i])

        for i in range(args.num_layers):
            seq = []
            neighbour_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i + 1] * self.edge_num[i]
            if i == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            last_pool = True if i == args.num_layers - 1 else False

            # (T, J, D) => (T, J', D)
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbour_list), last_pool=last_pool)

            if args.use_residual_blocks:
                # (T, J, D) => (T/2, J', 2D)
                seq.append(SkeletonResidual(self.topologies[i], neighbour_list, joint_num=self.edge_num[i], in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=2, padding=padding, padding_mode=args.padding_mode, bias=bias,
                                            extra_conv=args.extra_conv, pooling_mode=args.skeleton_pool, activation=args.activation, last_pool=last_pool))
            else:
                for _ in range(args.extra_conv):
                    # (T, J, D) => (T, J, D)
                    seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=in_channels,
                                            joint_num=self.edge_num[i], kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                                            stride=1,
                                            padding=padding, padding_mode=args.padding_mode, bias=bias))
                    seq.append(nn.PReLU() if args.activation == 'relu' else nn.Tanh())
                # (T, J, D) => (T/2, J, 2D)
                seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=out_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=False,
                                        in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))

                seq.append(pool)
                seq.append(nn.PReLU() if args.activation == 'relu' else nn.Tanh())
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

    def forward(self, input):
        output = input.permute(0, 2, 1)#input.reshape(bs, n, -1, 6)
        for layer in self.layers:
            output = layer(output)
        output = output.permute(0, 2, 1)
        return output


class WrapedMotionEncoder(nn.Module):
    def __init__(self, args, downsample=4):
        super(WrapedMotionEncoder, self).__init__()
        self.args = args
        
        self.downsample = downsample
        self.embed_dim = self.args.motion_f
        assert self.downsample in [2, 4], "downsample must be 2 or 4"
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = self.args.motion_f
        args_top.vae_test_dim = self.args.motion_dim
        self.feature_extractor = VQEncoderV6(args_top)
        
        self.encoder_cnn = Encoder1D(
            input_emb_width=self.args.motion_f,
            down_t=2 if self.downsample == 4 else 1,
            depth=3,
            dilation_growth_rate=3,
            activation='gelu',
            output_emb_width=self.args.motion_f,
        )
        
        self.pos_encoding = PositionalEncoding(d_model=self.args.motion_f)
        
        if self.args.n_layer >= 1:
            encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.args.motion_f,  # This should match the hidden size of the Wav2Vec2 model
            nhead=8,      # Number of attention heads
            dim_feedforward=self.args.hidden_size,  # The feedforward network dimension
            dropout=0.1,   # Dropout rate
            batch_first=True
            )
            self.encoder_trans = nn.TransformerEncoder(encoder_layer, num_layers=self.args.n_layer)

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
        
        # Filter for encoder-related keys
        for key, value in state_dict.items():
            if key.startswith('motion_encoder_fintune.'):
                # Remove the 'encoder.' prefix to match our state dict keys
                new_key = key[len('motion_encoder_fintune.'):]
                encoder_state_dict[new_key] = value
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = self.load_state_dict(encoder_state_dict, strict=strict)
        
        if strict:
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        return checkpoint.get('iteration', None)  # Return the iteration number if available

    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing for memory efficiency."""
        # For transformer encoder
        if self.args.n_layer >= 1:
            if hasattr(self.encoder_trans, 'layers'):
                for layer in self.encoder_trans.layers:
                    if enable:
                        # Store original forward
                        layer._forward = layer.forward
                        # Create wrapper that only passes the input tensor
                        def make_checkpoint_wrapper(module):
                            def wrapper(*args, **kwargs):
                                # Only pass the input tensor, no masks since they're not used
                                return torch.utils.checkpoint.checkpoint(
                                    module._forward,
                                    args[0],  # input tensor
                                    use_reentrant=False
                                )
                            return wrapper
                        layer.forward = make_checkpoint_wrapper(layer)
                    else:
                        if hasattr(layer, '_forward'):
                            layer.forward = layer._forward
                            delattr(layer, '_forward')

    def forward(self, inputs, separate_levels: bool = True):
        motion_feats = self.feature_extractor(inputs)
        low_level = self.encoder_cnn(motion_feats) # downsample of 4
        
        if self.args.n_layer >= 1:
            hidden_states = self.pos_encoding(low_level.detach() if separate_levels else low_level)
            hidden_states = self.encoder_trans(hidden_states)
        else:
            hidden_states = low_level
        
        return {
            "low_level": low_level,
            "high_level": hidden_states
        }

class WrapedMotionDecoder(nn.Module):
    def __init__(self, args, downsample=4):
        super(WrapedMotionDecoder, self).__init__()
       
        """Reverse the motion encoder.
        This class is used to decode the motion features extracted from the encoder.
        It uses a transformer decoder and a 1D convolutional decoder.
        The transformer decoder is used to decode the high-level features,
        and the 1D convolutional decoder is used to decode the low-level features.
        
        Args:
            args: Arguments containing the model configuration.
            downsample: The downsampling factor. Must be either 2 or 4.
        
        """
        self.args = args
        
        self.downsample = downsample
        assert self.downsample in [2, 4], "downsample must be 2 or 4"
        
        if self.args.n_layer >= 1:
            self.pos_encoding = PositionalEncoding(d_model=self.args.motion_f)
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=self.args.motion_f,  # This should match the hidden size of the Wav2Vec2 model
                nhead=8,      # Number of attention heads
                dim_feedforward=self.args.hidden_size,  # The feedforward network dimension
                dropout=0.1,   # Dropout rate
                batch_first=True
            )
            self.decoder_trans = nn.TransformerEncoder(decoder_layer, num_layers=self.args.n_layer)
        
        self.decoder_cnn = Decoder1D(
            input_emb_width=self.args.motion_f,
            down_t=2 if self.downsample == 4 else 1,
            depth=3,
            dilation_growth_rate=3,
            activation='gelu',
            output_emb_width=self.args.motion_f,
        )
        
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = self.args.motion_f
        args_top.vae_test_dim = self.args.motion_dim
        
        self.motion_reconstruct = VQDecoderV6(args_top)
    
    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing for memory efficiency."""
        # For transformer encoder
        if hasattr(self.encoder_trans, 'layers'):
            for layer in self.encoder_trans.layers:
                if enable:
                    # Store original forward
                    layer._forward = layer.forward
                    # Create wrapper that only passes the input tensor
                    def make_checkpoint_wrapper(module):
                        def wrapper(*args, **kwargs):
                            # Only pass the input tensor, no masks since they're not used
                            return torch.utils.checkpoint.checkpoint(
                                module._forward,
                                args[0],  # input tensor
                                use_reentrant=False
                            )
                        return wrapper
                    layer.forward = make_checkpoint_wrapper(layer)
                else:
                    if hasattr(layer, '_forward'):
                        layer.forward = layer._forward
                        delattr(layer, '_forward')
        
        # For CNN encoder
        for module in self.encoder_cnn.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                if enable:
                    module._forward = module.forward
                    module.forward = lambda *args, **kwargs: torch.utils.checkpoint.checkpoint(
                        module._forward, *args, use_reentrant=False
                    )
                else:
                    if hasattr(module, '_forward'):
                        module.forward = module._forward
                        delattr(module, '_forward')

        
    def forward(self, inputs):
        if self.args.n_layer >= 1:
            hidden_states = self.pos_encoding(inputs)
            hidden_states = self.decoder_trans(hidden_states) # (bs, n, 512)
        else:
            hidden_states = inputs
        
        low_level = self.decoder_cnn(hidden_states) # upsample of 4
        motion = self.motion_reconstruct(low_level)
        
        return {
            "motion": motion,
            "high_level": hidden_states,
            "low_level": low_level
        }



class WrapedMotionEncoderV2(nn.Module):
    def __init__(self, args, downsample=4):
        """
        Remove the feature extractor for this version.
        """
        super(WrapedMotionEncoderV2, self).__init__()
        self.args = args
        
        self.downsample = downsample
        self.embed_dim = self.args.motion_f
        assert self.downsample in [2, 4], "downsample must be 2 or 4"
        # args_top = copy.deepcopy(self.args)
        # args_top.vae_layer = 3
        # args_top.vae_length = self.args.motion_f
        # args_top.vae_test_dim = self.args.motion_dim
        # self.feature_extractor = VQEncoderV6(args_top)
        
        self.encoder_cnn = Encoder1D(
            input_emb_width=self.args.motion_dim,
            down_t=2 if self.downsample == 4 else 1,
            depth=3,
            dilation_growth_rate=3,
            activation='gelu',
            output_emb_width=self.args.motion_f,
        )
        
        self.pos_encoding = PositionalEncoding(d_model=self.args.motion_f)
        
        if self.args.n_layer >= 1:
            encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.args.motion_f,  # This should match the hidden size of the Wav2Vec2 model
            nhead=8,      # Number of attention heads
            dim_feedforward=self.args.hidden_size,  # The feedforward network dimension
            dropout=0.1,   # Dropout rate
            batch_first=True
            )
            self.encoder_trans = nn.TransformerEncoder(encoder_layer, num_layers=self.args.n_layer)

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
        
        # Filter for encoder-related keys
        for key, value in state_dict.items():
            if key.startswith('motion_encoder_fintune.'):
                # Remove the 'encoder.' prefix to match our state dict keys
                new_key = key[len('motion_encoder_fintune.'):]
                encoder_state_dict[new_key] = value
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = self.load_state_dict(encoder_state_dict, strict=strict)
        
        if strict:
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        return checkpoint.get('iteration', None)  # Return the iteration number if available

    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing for memory efficiency."""
        # For transformer encoder
        if self.args.n_layer >= 1:
            if hasattr(self.encoder_trans, 'layers'):
                for layer in self.encoder_trans.layers:
                    if enable:
                        # Store original forward
                        layer._forward = layer.forward
                        # Create wrapper that only passes the input tensor
                        def make_checkpoint_wrapper(module):
                            def wrapper(*args, **kwargs):
                                # Only pass the input tensor, no masks since they're not used
                                return torch.utils.checkpoint.checkpoint(
                                    module._forward,
                                    args[0],  # input tensor
                                    use_reentrant=False
                                )
                            return wrapper
                        layer.forward = make_checkpoint_wrapper(layer)
                    else:
                        if hasattr(layer, '_forward'):
                            layer.forward = layer._forward
                            delattr(layer, '_forward')

    def forward(self, inputs, separate_levels: bool = True):
        motion_feats = inputs
        low_level = self.encoder_cnn(motion_feats) # downsample of 4
        
        if self.args.n_layer >= 1:
            hidden_states = self.pos_encoding(low_level.detach() if separate_levels else low_level)
            hidden_states = self.encoder_trans(hidden_states)
        else:
            hidden_states = low_level
        
        return {
            "low_level": low_level,
            "high_level": hidden_states
        }

class WrapedMotionDecoderV2(nn.Module):
    def __init__(self, args, downsample=4):
        super(WrapedMotionDecoderV2, self).__init__()
        self.args = args
        self.downsample = downsample
        assert self.downsample in [2, 4], "downsample must be 2 or 4"

        self.decoder_cnn = Decoder1D(
            input_emb_width=self.args.motion_dim,
            down_t=2 if self.downsample == 4 else 1,
            depth=3,
            dilation_growth_rate=3,
            activation='gelu',
            output_emb_width=self.args.motion_f,
        )
        
        self.pos_encoding = PositionalEncoding(d_model=self.args.motion_f)
        
        if self.args.n_layer >= 1:
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=self.args.motion_f,  # This should match the hidden size of the Wav2Vec2 model
                nhead=8,      # Number of attention heads
                dim_feedforward=self.args.hidden_size,  # The feedforward network dimension
                dropout=0.1,   # Dropout rate
                batch_first=True
            )
            self.decoder_trans = nn.TransformerEncoder(decoder_layer, num_layers=self.args.n_layer)

    def forward(self, inputs):
        if self.args.n_layer >= 1:
            hidden_states = self.pos_encoding(inputs)
            hidden_states = self.decoder_trans(hidden_states)
        else:
            hidden_states = inputs
        
        motion = self.decoder_cnn(hidden_states) # upsample of 4
        
        return {
            "motion": motion,
            "high_level": hidden_states,
        }


class WrapedMotionEncoderV3(nn.Module):
    def __init__(self, args, downsample=4):
        """
        Remove the feature extractor for this version.
        """
        super(WrapedMotionEncoderV3, self).__init__()
        self.args = args
        
        self.downsample = downsample
        self.embed_dim = self.args.motion_f
        assert self.downsample in [2, 4], "downsample must be 2 or 4"
        
        self.stem = Stem(
            in_chs=self.args.motion_dim,
            out_chs=self.args.motion_f,
            act_layer='gelu',
            norm_layer='leakyrelu',
            leaky_relu_slope=0.2,
        )
        
        self.encoder_cnn = Encoder1D(
            input_emb_width=self.args.motion_f,
            down_t=2 if self.downsample == 4 else 1,
            depth=3,
            dilation_growth_rate=3,
            activation='gelu',
            output_emb_width=self.args.motion_f,
        )
        
        self.pos_encoding = PositionalEncoding(d_model=self.args.motion_f)
        
        if self.args.n_layer >= 1:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.args.motion_f,
                nhead=8,
                dim_feedforward=self.args.hidden_size,
                dropout=0.1,
                batch_first=True
            )
            self.encoder_trans = nn.TransformerEncoder(encoder_layer, num_layers=self.args.n_layer)

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
        
        # Filter for encoder-related keys
        for key, value in state_dict.items():
            if key.startswith('motion_encoder_fintune.'):
                # Remove the 'encoder.' prefix to match our state dict keys
                new_key = key[len('motion_encoder_fintune.'):]
                encoder_state_dict[new_key] = value
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = self.load_state_dict(encoder_state_dict, strict=strict)
        
        if strict:
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        return checkpoint.get('iteration', None)  # Return the iteration number if available

    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing for memory efficiency."""
        # For transformer encoder
        if self.args.n_layer >= 1:
            if hasattr(self.encoder_trans, 'layers'):
                for layer in self.encoder_trans.layers:
                    if enable:
                        # Store original forward
                        layer._forward = layer.forward
                        # Create wrapper that only passes the input tensor
                        def make_checkpoint_wrapper(module):
                            def wrapper(*args, **kwargs):
                                # Only pass the input tensor, no masks since they're not used
                                return torch.utils.checkpoint.checkpoint(
                                    module._forward,
                                    args[0],  # input tensor
                                    use_reentrant=False
                                )
                            return wrapper
                        layer.forward = make_checkpoint_wrapper(layer)
                    else:
                        if hasattr(layer, '_forward'):
                            layer.forward = layer._forward
                            delattr(layer, '_forward')

    def forward(self, inputs, separate_levels: bool = True):
        motion_feats = self.stem(inputs)
        low_level = self.encoder_cnn(motion_feats) # downsample of 4
        
        if self.args.n_layer >= 1:
            hidden_states = self.pos_encoding(low_level.detach() if separate_levels else low_level)
            hidden_states = self.encoder_trans(hidden_states)
        else:
            hidden_states = low_level
        
        return {
            "low_level": low_level,
            "high_level": hidden_states
        }

class WrapedMotionDecoderV3(nn.Module):
    def __init__(self, args, downsample=4):
        super(WrapedMotionDecoderV3, self).__init__()
        self.args = args
        self.downsample = downsample
        assert self.downsample in [2, 4], "downsample must be 2 or 4"

        self.motion_reconstruct = Stem(
            in_chs=self.args.motion_f,
            out_chs=self.args.motion_dim,
            act_layer='gelu',
            norm_layer='leakyrelu',
            leaky_relu_slope=0.2,
        )
        self.decoder_cnn = Decoder1D(
            input_emb_width=self.args.motion_f,
            down_t=2 if self.downsample == 4 else 1,
            depth=3,
            dilation_growth_rate=3,
            activation='gelu',
            output_emb_width=self.args.motion_f,
        )
        
        self.pos_encoding = PositionalEncoding(d_model=self.args.motion_f)
        
        if self.args.n_layer >= 1:
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=self.args.motion_f,  # This should match the hidden size of the Wav2Vec2 model
                nhead=8,      # Number of attention heads
                dim_feedforward=self.args.hidden_size,  # The feedforward network dimension
                dropout=0.1,   # Dropout rate
                batch_first=True
            )
            self.decoder_trans = nn.TransformerEncoder(decoder_layer, num_layers=self.args.n_layer)

    def forward(self, inputs):
        if self.args.n_layer >= 1:
            hidden_states = self.pos_encoding(inputs)
            hidden_states = self.decoder_trans(hidden_states)
        else:
            hidden_states = inputs
        
        low_level = self.decoder_cnn(hidden_states) # upsample of 4
        motion = self.motion_reconstruct(low_level)
        
        return {
            "motion": motion,
            "high_level": hidden_states,
        }