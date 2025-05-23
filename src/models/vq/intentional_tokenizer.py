import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


from models.vq.m_quantizer import VectorQuantizerM
from models.layers.motion_encoder import WrapedMotionEncoder, WrapedMotionDecoder, WrapedMotionEncoderV2, WrapedMotionDecoderV2, WrapedMotionEncoderV3, WrapedMotionDecoderV3
from models.vq.vqvae import AttnProjection


class IntentionalTokenizer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_query = args.tokenizer.num_query

        if getattr(args.tokenizer, 'v2', False):
            self.encoder = WrapedMotionEncoderV2(
                args.tokenizer,
            )
        elif getattr(args.tokenizer, 'v3', False):
            self.encoder = WrapedMotionEncoderV3(
                args.tokenizer,
            )
        else:
            self.encoder = WrapedMotionEncoder(
                args.tokenizer,
            )
        
        if args.tokenizer.quant_proj == 'linear':
            self.quant_proj = nn.Linear(self.encoder.embed_dim, args.tokenizer.vocab_width)
        elif args.tokenizer.quant_proj == 'attn':
            self.quant_proj = AttnProjection(self.encoder.embed_dim, args.tokenizer.vocab_width, self.encoder.embed_dim // args.tokenizer.vocab_width)
        else:
            raise NotImplementedError

        self.quantizer = VectorQuantizerM(
            vocab_size=args.tokenizer.vocab_size,
            vocab_width=args.tokenizer.vocab_width,
            beta=args.tokenizer.vq_beta,
            use_entropy_loss=args.tokenizer.le > 0,
            entropy_temp=args.tokenizer.e_temp,
            num_codebooks=args.tokenizer.num_codebooks,
            embed_proj=args.tokenizer.embed_proj,
        )

        if args.tokenizer.quant_proj == 'linear':
            self.post_quant_proj = nn.Linear(args.tokenizer.vocab_width, self.encoder.embed_dim)
        elif args.tokenizer.quant_proj == 'attn':
            self.post_quant_proj = AttnProjection(args.tokenizer.vocab_width, self.encoder.embed_dim, self.encoder.embed_dim // args.tokenizer.vocab_width)
        else:
            raise NotImplementedError

        if getattr(args.tokenizer, 'v2', False):
            self.decoder = WrapedMotionDecoderV2(
                args.tokenizer,
            )
        elif getattr(args.tokenizer, 'v3', False):
            self.decoder = WrapedMotionDecoderV3(
                args.tokenizer,
            )
        else:
            self.decoder = WrapedMotionDecoder(
                args.tokenizer,
            )

        # pretrained motion encoder
        pretrained_encoder_args = args.tokenizer.copy()
        pretrained_encoder_args.n_layer = args.data.n_layer
        
        if getattr(args.tokenizer, 'v2', False):
            self.pretrained_encoder = WrapedMotionEncoderV2(
                pretrained_encoder_args,
            )
        elif getattr(args.tokenizer, 'v3', False):
            self.pretrained_encoder = WrapedMotionEncoderV3(
                pretrained_encoder_args,
            )
        else:
            self.pretrained_encoder = WrapedMotionEncoder(
                pretrained_encoder_args,
            )
        
        # self.pretrained_encoder.load_from_pretrained(args.pretrained_encoder_path)
        
        self.fc_norm = nn.LayerNorm(self.encoder.embed_dim, eps=1e-6)
        self.projection = nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.maybe_record_function = nullcontext

        self.encoder.set_grad_checkpointing(args.grad_ckpt)
        self.pretrained_encoder.set_grad_checkpointing(args.grad_ckpt)

    def forward(self, motion, vae_bs=None, ret_usages=False):
        if vae_bs is None:
            vae_bs = motion.shape[0]
        motion_tokens = self.encoder(motion, separate_levels=False)
        motion_tokens = motion_tokens['high_level'].float()
        
        with torch.cuda.amp.autocast(enabled=False):
            motion_tokens = torch.utils.checkpoint.checkpoint(self.quant_proj, motion_tokens, use_reentrant=False)
            motion_tokens, vq_loss, entropy_loss, usages = self.quantizer(motion_tokens)
            motion_tokens = torch.utils.checkpoint.checkpoint(self.post_quant_proj, motion_tokens, use_reentrant=False)
        motion_rec = self.decoder(motion_tokens[:vae_bs])
        motion_rec = motion_rec['motion']

        motion_repr = motion_tokens.mean(dim=1) # [bs, embed_dim]
        motion_repr = self.projection(self.fc_norm(motion_repr))
        motion_repr = F.normalize(motion_repr, dim=-1)
        
        
        motion_pretrain = self.pretrained_encoder(motion)
        motion_pretrain = motion_pretrain['high_level']
        motion_pretrain = F.normalize(motion_pretrain, dim=-1)
        

        output_dict = {
            "motion_rec": motion_rec,
            "vq_loss": vq_loss,
            "entropy_loss": entropy_loss,
            "codebook_usages": usages,
            "motion_features": motion_repr,
            "motion_pretrain_features": motion_pretrain,
            "logit_scale": self.logit_scale.exp()
        }
        return output_dict
    
    def load_pretrained_encoder(self, pretrained_encoder_path):
        self.pretrained_encoder.load_from_pretrained(pretrained_encoder_path)

    def encode_motion(self, motion, normalize: bool = True, mean_pooling: bool = True):
        motion_tokens = self.encoder(motion, separate_levels=False)
        motion_tokens = motion_tokens['high_level'].float()
        motion_tokens = self.quant_proj(motion_tokens)
        motion_indices = self.quantizer.f_to_idx(motion_tokens)
        motion_tokens = self.quantizer.idx_to_f(motion_indices)
        motion_tokens = self.post_quant_proj(motion_tokens)
        if mean_pooling:
            features = motion_tokens.mean(dim=1)
        else:
            features = motion_tokens
        features = self.projection(self.fc_norm(features))
        return F.normalize(features, dim=-1) if normalize else features

    @torch.no_grad()
    def encode_clip_motion(self, motion, normalize: bool = True, mean_pooling: bool = True):
        features = self.pretrained_encoder(motion)
        features = features['high_level']
        if mean_pooling:
            features = features.mean(dim=1)
        else:
            features = features
        return F.normalize(features, dim=-1) if normalize else features

    def motion_to_idx(self, motion):
        motion_features = self.encoder(motion, separate_levels=False)
        motion_features = motion_features['high_level'].float()
        features = self.quant_proj(motion_features)
        return self.quantizer.f_to_idx(features)

    def idx_to_motion(self, indices):
        features = self.quantizer.idx_to_f(indices)
        features = self.post_quant_proj(features)
        motion = self.decoder(features)
        motion = motion['motion']
        return motion
    
    def semantic_loss(self, motion):
        rec_motion_features = self.encode_motion(motion)
        motion_features = self.encode_clip_motion(motion)
        cos_sim = F.cosine_similarity(rec_motion_features, motion_features)
        return F.relu(1 - 0.1 -cos_sim).mean() # use relu to avoid negative loss
    
    def local_semantic_loss(self, motion):
        rec_motion_features = self.encode_motion(motion, mean_pooling=False)
        motion_features = self.encode_clip_motion(motion, mean_pooling=False)
        cos_sim = F.cosine_similarity(rec_motion_features, motion_features)
        return F.relu(1 - 0.1 - cos_sim).mean() # use relu to avoid negative loss
    
    def map2postprojectedlatent(self, motion):
        """
        map motion to latent space after quantizer and post-projection
        """
        motion_tokens = self.encoder(motion, separate_levels=False)
        motion_tokens = motion_tokens['high_level'].float()
        motion_tokens = self.quant_proj(motion_tokens)
        motion_tokens, _, _, _ = self.quantizer(motion_tokens)
        motion_tokens = self.post_quant_proj(motion_tokens)
        return motion_tokens
    
    def postprojectedlatent2motion(self, latent):
        """
        map latent after quantizer and post-projection to motion space 
        """
        motion_rec = self.decoder(latent)
        return motion_rec['motion']
    
    def map2preprojectedlatent(self, motion):
        """
        map motion to latent space before quantizer and pre-projection
        """
        motion_tokens = self.encoder(motion, separate_levels=False)
        motion_tokens = motion_tokens['high_level'].float()
        return motion_tokens
    
    def preprojectedlatent2motion(self, latent):
        """
        map latent before pre-projection to motion space before quantizer
        """
        motion_tokens = self.quant_proj(latent)
        motion_tokens, _, _, _ = self.quantizer(motion_tokens)
        motion_tokens = self.post_quant_proj(motion_tokens)
        motion_rec = self.decoder(motion_tokens)
        return motion_rec['motion']
    
    def map2prequantizerlatent(self, motion):
        """
        map motion to latent space after pre-projection
        """
        motion_tokens = self.encoder(motion, separate_levels=False)
        motion_tokens = motion_tokens['high_level'].float()
        motion_tokens = self.quant_proj(motion_tokens)
        return motion_tokens
    
    def prequantizerlatent2motion(self, motion_tokens):
        """
        map pre-projected latent to motion space
        """
        motion_tokens, _, _, _ = self.quantizer(motion_tokens)
        motion_tokens = self.post_quant_proj(motion_tokens)
        motion_rec = self.decoder(motion_tokens)
        return motion_rec['motion']
    
    def map2postquantizerlatent(self, motion):
        """
        map motion to latent space after post-projection
        """
        motion_tokens = self.encoder(motion, separate_levels=False)
        motion_tokens = motion_tokens['high_level'].float()
        motion_tokens = self.quant_proj(motion_tokens)
        motion_tokens, _, _, _ = self.quantizer(motion_tokens)
        return motion_tokens
    
    def postquantizerlatent2motion(self, motion_tokens):
        """
        map post-projected latent to motion space
        """
        motion_tokens = self.post_quant_proj(motion_tokens)
        motion_rec = self.decoder(motion_tokens)
        return motion_rec['motion']

    def motion_to_reconstructed_motion(self, motion) -> torch.Tensor:
        motion_tokens = self.encoder(motion, separate_levels=False)
        motion_tokens = motion_tokens['high_level'].float()
        motion_tokens = self.quant_proj(motion_tokens)
        motion_tokens, _, _, _ = self.quantizer(motion_tokens)
        motion_tokens = self.post_quant_proj(motion_tokens)
        motion_rec = self.decoder(motion_tokens)
        motion_rec = motion_rec['motion']
        return motion_rec

    def lock_pretrain_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True, unlock_text_proj=False):
        self.pretrained_encoder.lock(unlocked_layers, freeze_layer_norm, unlock_text_proj)


if __name__ == '__main__':
    
    args = {
        "model": {
            "name_pyfile": "models.motion_encoder",
            "class_name": "MotionEncoder",
            "motion_f": 512,
            "audio_sr": 16000,
            "audio_fps": 16000,
            "audio_norm": False,
            "audio_f": 512,
            "word_rep": "textgrid",
            "word_index_num": 11195,
            "word_dims": 300,
            "facial_rep": "smplxflame_30",
            "facial_dims": 100,
            "facial_norm": False,
            "facial_f": 0,
            "f_pre_encoder": None,
            "f_encoder": None,
            "f_fix_pre": False,
            "id_rep": "onehot",
            "speaker_f": 0,
            "hidden_size": 768,
            "n_layer": 3,
            "motion_dim": 825
            
        },
        "pretrained_encoder_path": "./output",
        "vocab_size": 2048,
        "vq_beta": 0.25,
        "le": 0,
        "e_temp": 1.0,
        "num_codebooks": 4,
        "quant_proj": "attn",
        "num_query": 128,
        "grad_ckpt": False,
        "pretrained_encoder_path": "./output",
        "vocab_width": 64,
        
    }
    tokenizer = IntentionalTokenizer(args)
    
    motion = torch.randn(1, 128, 825) # 55 * 15 = 825
    output = tokenizer(motion, 1, ret_usages=True)
    print(output["codebook_usages"])
    print(output["vq_loss"])
    print(output["entropy_loss"])
    print(output["motion_features"])
    print(output["motion_pretrain_features"])
    print(output["logit_scale"])
    
