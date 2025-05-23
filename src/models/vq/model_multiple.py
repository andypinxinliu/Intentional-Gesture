import random
import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder, Encoder2d, Decoder2d
from models.vq.residual_vq import ResidualVQ
from models.gesture_clip import JointEmbedding
    
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 joint_representation=False,
                 semantic_encode=False,
                 semantic_dim=128,
                 residual=False,
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim

        self.code_dim = code_dim
        self.num_code = nb_code
        
        
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }

        if semantic_encode:
            self.semantic_encoder = JointEmbedding(args.clip_config)

            ### task layer ###
            self.encode_task_layer = nn.Sequential(
                nn.Linear(args.clip_config.embed_dim, args.clip_config.embed_dim),
                nn.Tanh(),
                nn.Linear(args.clip_config.embed_dim, semantic_dim) # for quantize
            )
            self.decode_task_layer = nn.Sequential(
                nn.Linear(args.clip_config.embed_dim, args.clip_config.embed_dim),
                nn.Tanh(),
                nn.Linear(args.clip_config.embed_dim, self.decoder_out_dim),
            )
            self.encode_task_layer.apply(self._init_weights)
            self.decode_task_layer.apply(self._init_weights)
            ### task layer ###

        if joint_representation:
            # the input will be of shape (bs, T, dim) where dim = 55 * 15

            self.upper_encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
            self.hand_encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
            self.lower_encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
            self.decoder = Decoder2d(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
        
        else:
            # the input will be of shape (bs, T, )



        
        
        if self.residual:
            self.quantizer = ResidualVQ(**rvqvae_config)
        
        

        if semantic_encode:
            
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)

        return code_idx, all_codes

    def forward(self, x):
        
        if type(x) is tuple:
            upper, lower, hand = x
            upper = self.preprocess(upper)
            lower = self.preprocess(lower)
            hand = self.preprocess(hand)
        else:
            raise ValueError('Input should be tuple of (upper, lower, hand)')
        
        # Encode
        upper_encoder = self.upper_encoder(upper)
        hand_encoder = self.hand_encoder(hand)
        lower_encoder = self.lower_encoder(lower)

        x_encoder = torch.cat([upper_encoder, hand_encoder, lower_encoder], dim=1)
        
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return  {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }


    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out
    
    def map2latent(self,x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0,2,1)
        return x_encoder

    def latent2origin(self,x):
        x = x.permute(0,2,1)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity


class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)