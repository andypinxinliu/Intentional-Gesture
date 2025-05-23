import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint

def get_norm_layer(norm_type):
    if norm_type == 'layernorm':
        return nn.LayerNorm
    elif norm_type == 'groupnorm':
        return nn.GroupNorm
    elif norm_type == 'batchnorm':
        return nn.BatchNorm1d
    elif norm_type == 'leakyrelu':
        return nn.LeakyReLU
    else:
        raise NotImplementedError(f"Normalization layer {norm_type} not implemented")

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words=11195, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, word_cache=False):
        super(TextEncoderTCN, self).__init__()

        num_channels = [args.hidden_size] #* args.n_layer
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], args.word_f)
        self.drop = nn.Dropout(emb_dropout)
        #self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        y = self.tcn(input.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y, torch.max(y, dim=1)[0]


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2
    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)
    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )
    return net

class BasicBlock(nn.Module):
    """ based on timm: https://github.com/rwightman/pytorch-image-models """
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU,   norm_layer=nn.BatchNorm1d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,  stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes), 
            )
        else: self.downsample=None
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x
            
class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class nonlinearity(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=0.2):
        super(ResConv1DBlock, self).__init__()

        padding = dilation
        self.norm = norm

        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()

        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Stem(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            act_layer: str = 'gelu',
            norm_layer: str = 'leakyrelu',
            leaky_relu_slope: float = 0.2,
            bias: bool = True,
    ):
        super().__init__()
        self.grad_checkpointing=False
        norm_act_layer = partial(get_norm_layer(norm_layer), leaky_relu_slope)
        self.out_chs = out_chs
        self.conv1 = nn.Conv1d(in_chs, out_chs, kernel_size=3, stride=1, padding=1)
        self.norm1 = norm_act_layer(out_chs)
        self.conv2 = nn.Conv1d(out_chs, out_chs, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.grad_checkpointing:
            x = checkpoint(self.conv1, x)
            x = self.norm1(x)
            x = checkpoint(self.conv2, x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.conv2(x)
        x = x.transpose(1, 2)
        return x
        

class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        act_layer=None,
        drop=0.0,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer('layernorm'), eps=1e-6)
        self.norm = norm_layer(in_features)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False, 
                 norm_first=False, device=None, dtype=None):
        
        super().__init__(d_model, nhead, dim_feedforward, dropout, 
                        activation, layer_norm_eps, batch_first, 
                        norm_first, device, dtype)
        
        # Replace the feedforward network with our custom GeGluMlp
        self.linear1 = None
        self.linear2 = None
        
        # Create our custom GeGluMlp
        self.geglu_mlp = GeGluMlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            drop=dropout
        )
        
    def _ff_block(self, x):
        # Override the feedforward block to use our GeGluMlp
        return self.geglu_mlp(x)