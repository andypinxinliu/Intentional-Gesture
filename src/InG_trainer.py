import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
from typing import Dict
from utils import config, logger_tools, other_tools, metric, data_transfer, other_tools_hf
from dataloaders import data_tools
import librosa
from models.vq.intentional_tokenizer import IntentionalTokenizer
import wandb
import math
from tqdm import tqdm


def convert_15d_to_6d(motion):
    """
    Convert 15D motion to 6D motion, the current motion is 15D, but the eval model is 6D
    """
    bs = motion.shape[0]
    motion_6d = motion.reshape(bs, -1, 55, 15)[:, :, :, 6:12]
    motion_6d = motion_6d.reshape(bs, -1, 55*6)
    return motion_6d


class CustomTrainer(train.BaseTrainer):
    '''
    Multi-Modal AutoEncoder
    '''
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        self.joints = 55


        self.tracker = other_tools.EpochTracker(["fgd", "test_clip_fgd", "bc", "l1div", "predict_x0_loss"], [True,True,True,True,True])
            
        
        ##### Model #####

        model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["something"])
        
        if self.cfg.ddp:
            self.model = getattr(model_module, cfg.model.g_name)(cfg).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=False)
        else: 
            self.model = torch.nn.DataParallel(getattr(model_module, cfg.model.g_name)(cfg), self.cfg.gpus).cuda()
        
        # resume the audio encoder
        try:
            self.model.module.modality_encoder.load_from_pretrained(self.cfg.pretrained_encoder_path)
        except:
            logger.info(f"initializing audio encoder from scratch")
        
        
        if self.args.mode == "train":
            if self.rank == 0:
                logger.info(self.model)
                logger.info(f"init {self.cfg.model.g_name} success")
                wandb.watch(self.model)
            
        
        ##### Optimizer and Scheduler #####
        # Replace custom optimizer and scheduler with AdamW and CosineAnnealingLR
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.solver.learning_rate,
            weight_decay=self.cfg.solver.weight_decay if hasattr(self.cfg.solver, 'weight_decay') else 0.01,
            betas=(0.9, 0.95)
        )
        
        self.opt_s = torch.optim.lr_scheduler.LinearLR(
            self.opt,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.cfg.solver.epochs  # Total number of epochs
        )
        

        ##### VQ-VAE models #####
        """Initialize and load VQ-VAE models for different body parts."""
        # Body part VQ models
        self.vq_model = IntentionalTokenizer(self.cfg)

        # Load and filter state dict to exclude pretrained encoder parameters
        state_dict = torch.load(self.cfg.vq_path)['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if not k.startswith('pretrained_encoder')}

        # Load the filtered state dict with strict=False to allow missing keys
        self.vq_model.load_state_dict(filtered_state_dict, strict=False)
        self.vq_model.eval().to(self.rank)
        
        self.latent_type = getattr(self.cfg, 'latent_type', 'postprojected')
        
        ##### Loss functions #####
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        
        
        ##### Normalization #####
        self.mean_pose = torch.from_numpy(np.load(self.cfg.mean_pose_path)).to(self.rank)
        self.std_pose = torch.from_numpy(np.load(self.cfg.std_pose_path)).to(self.rank)
        
        
        if self.args.checkpoint:
            ckpt_state_dict = torch.load(self.args.checkpoint)['model_state_dict']
            # remove 'audioEncoder' from the state_dict due to legacy issues
            ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if 'modality_encoder.audio_encoder.' not in k}
            self.model.load_state_dict(ckpt_state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {self.args.checkpoint}")

    
    def _load_data(self, dict_data):
        audio_name = dict_data["audio_name"]
        facial_rep = dict_data["facial"].to(self.rank)
        beta = dict_data["beta"].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank)
        cached_rep15d = dict_data["rep15d"].to(self.rank)
        tar_pose = convert_15d_to_6d(dict_data["rep15d"]).to(self.rank)
        cached_rep15d = (cached_rep15d - self.mean_pose) / self.std_pose
        cached_audio_low = dict_data["audio_low"].to(self.rank) # (bs, T, C) C = 512
        cached_audio_high = dict_data["audio_high"].to(self.rank) # (bs, T, C) C = 768
        bert_time_aligned = dict_data["bert_time_aligned"].to(self.rank) # (bs, T, C) C = 768
        
        intention_embeddings = dict_data.get("intention_embeddings", None)
        if intention_embeddings is not None:
            intention_embeddings = intention_embeddings.to(self.rank)
        
        intention_mask = dict_data.get("intention_mask", None)
        if intention_mask is not None:
            intention_mask = intention_mask.to(self.rank)
        
        audio_onset = None  
        if self.cfg.data.onset_rep:
            audio_onset = dict_data["audio_onset"].to(self.rank)
            
        word = dict_data.get("word", None)
        if word is not None:
            word = word.to(self.rank)
            
        # used during testing for time segment
        intention_timings = dict_data.get("intention_timings", None)
        intention_texts = dict_data.get("intention_texts", None)
        intention_embedding_lengths = dict_data.get("intention_embedding_lengths", None)
        
        cached_audio_high = torch.cat([cached_audio_high, bert_time_aligned], dim=-1) # [bs, T, C] C = 1536 (768 + 768)
        
        audio_tensor = dict_data["audio_tensor"].to(self.rank) # (bs, T) T = 68266
        
        #TODO: I need to test whether postquantizer or prequantizer is better
        # default is postprojected
        if self.latent_type == 'prequantizer':
            latent_in = self.vq_model.map2prequantizerlatent(cached_rep15d)
        elif self.latent_type == 'postquantizer':
            latent_in = self.vq_model.map2postquantizerlatent(cached_rep15d)
        elif self.latent_type == 'preprojected':
            latent_in = self.vq_model.map2preprojectedlatent(cached_rep15d)
        elif self.latent_type == 'postprojected':
            latent_in = self.vq_model.map2postprojectedlatent(cached_rep15d)
        
        # style feature is always None (without annotation, we never know what it is)
        style_feature = None
        
        return {
            "cached_audio_low": cached_audio_low,
            "audio_tensor": audio_tensor,
            "cached_audio_high": cached_audio_high,
            "audio_onset": audio_onset,
            "word": word,
            "latent_in":  latent_in,
            "tar_id": tar_id,
            "facial_rep": facial_rep,
            "beta": beta,
            "tar_pose": tar_pose,
            "trans": tar_trans,
            "style_feature": style_feature,
            "intention_embeddings": intention_embeddings,
            "intention_mask": intention_mask,
            "intention_timings": intention_timings,
            "intention_texts": intention_texts,
            "intention_embedding_lengths": intention_embedding_lengths,
            "audio_name": audio_name,
        }
    
    def _g_training(self, loaded_data, mode="train", epoch=0):
            
        cond_ = {'y':{}}
        cond_['y']['audio_tensor'] = loaded_data['audio_tensor']
        cond_['y']['audio_low'] = loaded_data['cached_audio_low']
        cond_['y']['audio_high'] = loaded_data['cached_audio_high']
        cond_['y']['audio_onset'] = loaded_data['audio_onset']
        cond_['y']['word'] = loaded_data['word']
        cond_['y']['id'] = loaded_data['tar_id']
        cond_['y']['seed'] = loaded_data['latent_in'][:,:self.cfg.pre_frames]
        cond_['y']['style_feature'] = loaded_data['style_feature']
        cond_['y']['intention_embeddings'] = loaded_data['intention_embeddings']
        cond_['y']['intention_mask'] = loaded_data['intention_mask']
        x0 = loaded_data['latent_in']
        x0 = x0.permute(0, 2, 1).unsqueeze(2)

        g_loss_final = self.model.module.train_forward(cond_, x0, train_consistency=True)['loss']

        self.tracker.update_meter("predict_x0_loss", "train", g_loss_final.item())

        if mode == 'train':
            return g_loss_final
    
    def _g_training_reflow(self, loaded_data, mode="train", epoch=0):
        
        latents = loaded_data['sample'].squeeze(0)
        at_feat = loaded_data['at_feat'].squeeze(0)
        noise = loaded_data['noise'].squeeze(0)
        seed = loaded_data['seed'].squeeze(0)
        
        g_loss_final = self.model.module.train_reflow(latents, at_feat, noise, seed)['loss']
        self.tracker.update_meter("predict_x0_loss", "train", g_loss_final.item())  
        
        if mode == 'train':
            return g_loss_final

    def _get_relevant_intentions(self, segment_start_time, segment_duration, intention_timings, intention_embeddings, intention_lengths):
        """
        Find intentions that overlap significantly (>70%) with the current segment
        Args:
            segment_start_time: start time of current segment in seconds
            segment_duration: duration of current segment in seconds
            intention_timings: list of dicts with start_time and end_time
            intention_embeddings: tensor of intention embeddings
            intention_lengths: list of embedding lengths
        Returns:
            relevant_embeddings: tensor of intention embeddings or zeros
            relevant_mask: boolean mask for the embeddings
        """
        segment_end_time = segment_start_time + segment_duration
        relevant_indices = []
        
        # For each intention, check overlap percentage
        for idx, timing in enumerate(intention_timings[0]):  # assuming timing is nested list
            intention_start = timing['start_time']
            intention_end = timing['end_time']
            intention_duration = intention_end - intention_start
            
            # Calculate overlap
            overlap_start = max(segment_start_time, intention_start)
            overlap_end = min(segment_end_time, intention_end)
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                overlap_percentage = overlap_duration / intention_duration
                
                if overlap_percentage >= 0.7:  # 50% overlap threshold
                    relevant_indices.append(idx)
        
        # If we found relevant intentions, gather their embeddings
        # ablation: always set the relevant embeddings to zeros
        relevant_indices = []
        if relevant_indices:
            # Calculate start index for slicing embeddings
            start_idx = 0
            relevant_embeddings = []
            
            for idx in relevant_indices:
                if idx > 0:
                    start_idx = sum(intention_lengths[0][:idx])
                length = intention_lengths[0][idx]
                embedding = intention_embeddings[:, start_idx:start_idx + length]
                relevant_embeddings.append(embedding)
            
            # Concatenate all relevant embeddings
            if len(relevant_embeddings) > 0:
                relevant_embeddings = torch.cat(relevant_embeddings, dim=1)
                relevant_mask = torch.ones((intention_embeddings.shape[0], relevant_embeddings.shape[1]), 
                                         dtype=torch.bool, 
                                         device=intention_embeddings.device)
                return relevant_embeddings, relevant_mask
        
        # If no relevant intentions found, return zeros with zero mask
        # Use the same hidden dimension as the original embeddings
        hidden_dim = intention_embeddings.shape[-1]
        batch_size = intention_embeddings.shape[0]
        
        # Create zero embeddings with shape [batch_size, 1, hidden_dim]
        zero_embeddings = torch.zeros((batch_size, 1, hidden_dim), 
                                    device=intention_embeddings.device)
        
        # Create zero mask with shape [batch_size, 1]
        zero_mask = torch.zeros((batch_size, 1), 
                              dtype=torch.bool,
                              device=intention_embeddings.device)
        
        return zero_embeddings, zero_mask

    def _g_test(self, loaded_data):
        
        tar_id = loaded_data["tar_id"]
        style_feature = loaded_data["style_feature"]
        tar_beta = loaded_data["beta"]
        tar_pose = loaded_data["tar_pose"]
        tar_exps = loaded_data["facial_rep"]
        tar_trans = loaded_data["trans"]
        
        
        latent_in = loaded_data["latent_in"]
        in_audio = loaded_data["audio_tensor"]
        audio_onset = loaded_data["audio_onset"]
        in_word = loaded_data["word"]
        in_audio_high = loaded_data["cached_audio_high"]
        in_audio_low = loaded_data["cached_audio_low"]
        intention_embeddings = loaded_data["intention_embeddings"]
        intention_mask = loaded_data["intention_mask"]
        intention_timings = loaded_data["intention_timings"]
        intention_texts = loaded_data["intention_texts"]
        intention_embedding_lengths = loaded_data["intention_embedding_lengths"]

        in_x0 = loaded_data["latent_in"]
        in_seed = loaded_data["latent_in"]
        
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints
        
        remain = n%8
        if remain != 0:
            
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_exps = tar_exps[:, :-remain, :]
            in_x0 = in_x0[:, :in_x0.shape[1]-(remain//self.cfg.vqvae_squeeze_scale), :]
            in_seed = in_seed[:, :in_x0.shape[1]-(remain//self.cfg.vqvae_squeeze_scale), :]
            if in_word is not None:
                in_word = in_word[:, :-remain]
            n = n - remain

        rec_all = []
        vqvae_squeeze_scale = self.cfg.vqvae_squeeze_scale
        pre_frames_scaled = self.cfg.pre_frames * vqvae_squeeze_scale
        roundt = (n - pre_frames_scaled) // (self.cfg.data.pose_length - pre_frames_scaled)
        remain = (n - pre_frames_scaled) % (self.cfg.data.pose_length - pre_frames_scaled)
        round_l = self.cfg.pose_length - pre_frames_scaled
        round_audio = int(round_l / 3 * 5)
        pre_frames_scaled_audio = math.ceil(pre_frames_scaled / 3 * 5)
        
        in_audio_onset_tmp = None
        in_word_tmp = None
        for i in range(0, roundt):
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*pre_frames_scaled]
            if audio_onset is not None:
                in_audio_onset_tmp = audio_onset[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.cfg.pre_frames * vqvae_squeeze_scale]
            if in_word is not None:
                in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.cfg.pre_frames * vqvae_squeeze_scale]
            in_audio_high_tmp = in_audio_high[:, i*(round_audio):(i+1)*(round_audio) + pre_frames_scaled_audio]
            in_audio_low_tmp = in_audio_low[:, i*(round_audio):(i+1)*(round_audio)+ pre_frames_scaled_audio]
            
            
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.cfg.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.cfg.pre_frames]
            
            if i == 0:
                in_seed_tmp = in_seed_tmp[:, :self.cfg.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.cfg.pre_frames:, :]
            
            # Calculate current segment time
            segment_start_time = (i * round_l) / 30.0  # Convert frames to seconds
            segment_duration = round_l / 30.0
            
            # Get relevant intentions for this segment
            relevant_embeddings, relevant_mask = self._get_relevant_intentions(
                segment_start_time,
                segment_duration,
                loaded_data.get('intention_timings', [[]]),  # Assuming this is available in loaded_data
                loaded_data['intention_embeddings'],
                loaded_data.get('intention_embedding_lengths', [[]])  # Assuming this is available
            )
            
            cond_ = {'y':{}}
            cond_['y']['audio_tensor'] = in_audio_tmp
            cond_['y']['audio_low'] = in_audio_low_tmp
            cond_['y']['audio_high'] = in_audio_high_tmp
            cond_['y']['audio_onset'] = in_audio_onset_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] = in_seed_tmp
            cond_['y']['style_feature'] = torch.zeros([bs, 512]).cuda()
            cond_['y']['intention_embeddings'] = relevant_embeddings
            cond_['y']['intention_mask'] = relevant_mask
            
            sample = self.model(cond_)['latents']
            
            sample = sample.squeeze(2).permute(0, 2, 1)

            last_sample = sample.clone()
            
            rec_latent = sample
            
           
            if i == 0:
                rec_all.append(rec_latent)
            else:
                rec_all.append(rec_latent[:, self.cfg.pre_frames:])
        
        rec_all = torch.cat(rec_all, dim=1)
        
        
        if self.latent_type == 'prequantizer':
            rec_all = self.vq_model.prequantizerlatent2motion(rec_all)
        elif self.latent_type == 'postquantizer':
            rec_all = self.vq_model.postquantizerlatent2motion(rec_all)
        elif self.latent_type == 'preprojected':
            rec_all = self.vq_model.preprojectedlatent2motion(rec_all)
        elif self.latent_type == 'postprojected':
            rec_all = self.vq_model.postprojectedlatent2motion(rec_all)
        
        rec_pose = rec_all * self.std_pose + self.mean_pose
        
        # convert 15d to 6d
        rec_pose = convert_15d_to_6d(rec_pose)

        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        if self.cfg.model.use_exp:
            rec_exps = rec_face
        else:
            rec_exps = tar_exps
        
        rec_trans = tar_trans
        
        
        return {
            'rec_pose': rec_pose,
            'rec_exps': rec_exps,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,

        }
    
    def train(self, epoch):

        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, 'train', epoch)

            g_loss_final.backward()
            if self.cfg.solver.max_grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.solver.max_grad_norm)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']

            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.cfg.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.cfg.debug:
                if its == 1: 
                    break
        self.opt_s.step(epoch)

    def train_reflow(self, epoch):
        
        self.model.train()
        t_start = time.time()
        self.tracker.reset()

        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data_for_reflow(batch_data)
            t_data = time.time() - t_start
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training_reflow(loaded_data, 'train', epoch)

            g_loss_final.backward()
            if self.cfg.solver.max_grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.solver.max_grad_norm)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']

            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.cfg.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.cfg.debug:
                if its == 1: 
                    break
        
        self.opt_s.step(epoch)
    

    def val(self, epoch):
        

        self.tracker.reset()
        
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                bs = batch_data['rep15d'].shape[0]
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.cfg.data.pose_fps) != 1:
                    assert 30%self.cfg.data.pose_fps == 0
                    n *= int(30/self.cfg.data.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.cfg.data.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.cfg.data.pose_fps, mode='linear').permute(0,2,1)
                

                remain = n%self.cfg.vae_test_len
                
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.cfg.vae_length).detach().cpu().numpy()) # bs * n/8 * 240
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.cfg.vae_length).detach().cpu().numpy())
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )
                
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(bs, n, 127*3)[0, :n, :55*3]
                
                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    in_audio_eval, sr = librosa.load(self.cfg.data.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                    in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.cfg.data.audio_sr)
                    a_offset = int(self.align_mask * (self.cfg.data.audio_sr / self.cfg.data.pose_fps))
                    onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.cfg.data.audio_sr / self.cfg.data.pose_fps*n)], a_offset, len(in_audio_eval)-a_offset, True)
                    beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n-self.align_mask, 30, True)
                    align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n-2*self.align_mask))

                total_length += n
                

        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        
        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)
        
        align_avg = align/(total_length-2*len(self.test_loader)*self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.tracker.update_meter("bc", "val", align_avg)
        
        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)
        
        self.val_recording(epoch)

        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion")
    
    
    
    def test_clip(self, epoch):
        

        self.tracker.reset()
        
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(tqdm(self.test_clip_loader, desc="Testing CLIP", leave=True)):
                bs = batch_data['rep15d'].shape[0]
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.cfg.data.pose_fps) != 1:
                    assert 30%self.cfg.data.pose_fps == 0
                    n *= int(30/self.cfg.data.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.cfg.data.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.cfg.data.pose_fps, mode='linear').permute(0,2,1)
                

                remain = n%self.cfg.vae_test_len
                
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.cfg.vae_length).detach().cpu().numpy()) # bs * n/8 * 240
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.cfg.vae_length).detach().cpu().numpy())
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )
                
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(bs, n, 127*3)[0, :n, :55*3]
                
                _ = self.l1_calculator.run(joints_rec)
                
                total_length += n

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        
        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"test_clip fgd score: {fgd}")
        self.tracker.update_meter("test_clip_fgd", "val", fgd)
        
        l1div = self.l1_calculator.avg()
        logger.info(f"test_clip l1div score: {l1div}")
        # self.tracker.update_meter("test_clip_l1div", "val", l1div)
        
        current_time = time.time()
        test_clip_time = current_time - start_time
        logger.info(f"total test_clip inference time: {int(test_clip_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion")
        
        
        
        ############## Do the test data recording ##############
        latent_out = []
        latent_ori = []
        with torch.no_grad():
            for its, batch_data in enumerate(tqdm(self.test_loader, desc="Testing", leave=True)):
                bs = batch_data['rep15d'].shape[0]
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.cfg.data.pose_fps) != 1:
                    assert 30%self.cfg.data.pose_fps == 0
                    n *= int(30/self.cfg.data.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.cfg.data.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.cfg.data.pose_fps, mode='linear').permute(0,2,1)
                

                remain = n%self.cfg.vae_test_len
                
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.cfg.vae_length).detach().cpu().numpy()) # bs * n/8 * 240
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.cfg.vae_length).detach().cpu().numpy())
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )
                
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(bs, n, 127*3)[0, :n, :55*3]
                
                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    if bs == 1:
                        in_audio_eval, sr = librosa.load(self.cfg.data.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                        in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.cfg.data.audio_sr)
                        a_offset = int(self.align_mask * (self.cfg.data.audio_sr / self.cfg.data.pose_fps))
                        onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.cfg.data.audio_sr / self.cfg.data.pose_fps*n)], a_offset, len(in_audio_eval)-a_offset, True)
                        beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n-self.align_mask, 30, True)
                        align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n-2*self.align_mask))
                    
                    else:
                        align = 0 # if bs > 1, unable to calculate alignment, because the data sample are not aligned with original full audio seq

                total_length += n

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        
        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.tracker.update_meter("fgd", "val", fgd)
        
        align_avg = align/(total_length-2*len(self.test_loader)*self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.tracker.update_meter("bc", "val", align_avg)
        
        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.tracker.update_meter("l1div", "val", l1div)
                
        self.val_recording(epoch)

        end_time = time.time() - current_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion")
            
    
    def test(self, epoch):
        
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        # if os.path.exists(results_save_path): 
        #     print(f"Results already exist for epoch {epoch}, skipping test...")
        #     return 0
        os.makedirs(results_save_path, exist_ok=True)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        with torch.no_grad():
            
            for its, batch_data in enumerate(tqdm(self.test_loader, desc="Testing", leave=True)):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.cfg.data.pose_fps) != 1:
                    assert 30%self.cfg.data.pose_fps == 0
                    n *= int(30/self.cfg.data.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.cfg.data.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.cfg.pose_fps, mode='linear').permute(0,2,1)
                

                remain = n%self.cfg.vae_test_len
                
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.cfg.vae_length).detach().cpu().numpy()) # bs * n/8 * 240
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.cfg.vae_length).detach().cpu().numpy())
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )

                vertices_rec_face = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=rec_exps.reshape(bs*n, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3]-rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3]-rec_pose[:,3:21*3+3],
                        left_hand_pose=rec_pose[:,25*3:40*3]-rec_pose[:,25*3:40*3],
                        right_hand_pose=rec_pose[:,40*3:55*3]-rec_pose[:,40*3:55*3],
                        return_verts=True, 
                        return_joints=True,
                        leye_pose=rec_pose[:, 69:72]-rec_pose[:, 69:72],
                        reye_pose=rec_pose[:, 72:75]-rec_pose[:, 72:75],
                    )
                vertices_tar_face = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3]-tar_pose[:,:3],
                    body_pose=tar_pose[:,3:21*3+3]-tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3]-tar_pose[:,25*3:40*3],
                    right_hand_pose=tar_pose[:,40*3:55*3]-tar_pose[:,40*3:55*3],
                    return_verts=True, 
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72]-tar_pose[:, 69:72],
                    reye_pose=tar_pose[:, 72:75]-tar_pose[:, 72:75],
                )  
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                # joints_tar = vertices_tar["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                facial_rec = vertices_rec_face['vertices'].reshape(1, n, -1)[0, :n]
                facial_tar = vertices_tar_face['vertices'].reshape(1, n, -1)[0, :n]
                face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
                l2 = self.reclatent_loss(facial_rec, facial_tar)
                l2_all += l2.item() * n
                lvel += face_vel_loss.item() * n
                
                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    in_audio_eval, sr = librosa.load(self.cfg.data.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                    in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.cfg.data.audio_sr)
                    a_offset = int(self.align_mask * (self.cfg.data.audio_sr / self.cfg.data.pose_fps))
                    onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.cfg.data.audio_sr / self.cfg.data.pose_fps*n)], a_offset, len(in_audio_eval)-a_offset, True)
                    beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n-self.align_mask, 30, True)
                    align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n-2*self.align_mask))
                 
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load(self.cfg.data.data_path+self.cfg.data.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n
                
                # render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                #     results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz', 
                #     # results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz', 
                #     results_save_path,
                #     audio_name,
                #     self.args.data_path_1+"smplx_models/",
                #     use_matplotlib = False,
                #     args = self.args,
                #     )

        logger.info(f"l2 loss: {l2_all/total_length:.10f}")
        logger.info(f"lvel loss: {lvel/total_length:.10f}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fgd = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd score: {fgd}")
        self.test_recording("fgd", fgd, epoch) 
        
        align_avg = align/(total_length-2*len(self.test_loader)*self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)

        #data_tools.result2target_vis(self.cfg.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.cfg.data.pose_fps)} s motion")

    
    def test_render(self, epoch):

        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'

        '''
        input audio and text, output motion
        do not calculate loss and metric
        save video
        '''
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        os.makedirs(results_save_path, exist_ok=True)
        start_time = time.time()
        total_length = 0
        self.model.eval()
        self.smplx.eval()
        # self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                audio_name = loaded_data['audio_name'][0]
                
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.cfg.data.pose_fps) != 1:
                    assert 30%self.cfg.data.pose_fps == 0
                    n *= int(30/self.cfg.data.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.cfg.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.cfg.pose_fps, mode='linear').permute(0,2,1)
                

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                

                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load("./demo/examples/2_scott_0_1_1.npz", allow_pickle=True)
                
                file_name = audio_name.split("/")[-1].split(".")[0]
                results_npz_file_save_path = results_save_path+f"result_{file_name}"+'.npz'
                np.savez(results_npz_file_save_path,
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n
                
                render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                    results_npz_file_save_path, 
                    # results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz', 
                    results_save_path,
                    audio_name,
                    self.cfg.data.data_path_1+"smplx_models/",
                    use_matplotlib = False,
                    args = self.cfg,
                    )

    def load_checkpoint(self, checkpoint):
        # checkpoint is already a dict, do NOT call torch.load again!
        ckpt_state_dict = checkpoint['model_state_dict']
        # remove 'audioEncoder' from the state_dict due to legacy issues
        ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if 'modality_encoder.audio_encoder.' not in k}
        self.model.load_state_dict(ckpt_state_dict, strict=False)
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            self.opt_s.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'val_best' in checkpoint:
            self.val_best = checkpoint['val_best']
        logger.info("Checkpoint loaded successfully.")