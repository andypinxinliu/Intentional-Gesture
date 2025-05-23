import json
import torch
from torch.utils import data
import numpy as np
import librosa
import textgrid as tg
import os
import math


class BEAT2Dataset(data.Dataset):
    def __init__(self, cfg, split):
        data_meta_paths = cfg.data.meta_paths
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = [item for item in vid_meta if item.get("mode") == split]
        self.mean = 0  # np.load(cfg.data.mean_path) if cfg.data.mean_path is not None else 0
        self.std = 1  # np.load(cfg.data.std_path) if cfg.data.std_path is not None else 1
        self.joint_mask = None  # cfg.data.joint_mask if cfg.data.joint_mask is not None else None
        self.data_list = self.vid_meta
        self.fps = cfg.data.pose_fps
        self.audio_sr = cfg.data.audio_sr
        self.use_text = False  # cfg.data.use_text

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def normalize(motion, mean, std):
        return (motion - mean) / (std + 1e-7)

    @staticmethod
    def inverse_normalize(motion, mean, std):
        return motion * std + mean

    @staticmethod
    def select_joints(motion, joint_mask):
        return motion[:, joint_mask]

    @staticmethod
    def unselect_joints(motion, joint_mask):
        # for visualization
        full_motion = np.zeros((motion.shape[0], joint_mask.shape[0]))
        full_motion[:, joint_mask] = motion

    def __getitem__(self, item):
        data = self.data_list[item]
        motion = np.load(os.path.join(data["video_path"], data["video_id"] + ".npy"))
        sdx = data["start_idx"]
        edx = data["end_idx"]

        SMPLX_FPS = 30
        motion = motion[sdx:edx]
        audio = np.load(os.path.join(data["audio_path"], data["video_id"] + "_text.npz"), allow_pickle=True)
        sdx_audio = math.floor(sdx * (1 / SMPLX_FPS * 50))
        edx_audio = sdx_audio + int((edx - sdx) * 50 / SMPLX_FPS) + 1
        cached_audio_low = audio["wav2vec2_low"][sdx_audio:edx_audio]
        cached_audio_high = audio["wav2vec2_high"][sdx_audio:edx_audio]
        bert_time_aligned = audio["bert_time_aligned"][sdx_audio:edx_audio]

        motion_tensor = torch.from_numpy(motion).float()  # T x D
        cached_audio_low = torch.from_numpy(cached_audio_low).float()
        cached_audio_high = torch.from_numpy(cached_audio_high).float()
        bert_time_aligned = torch.from_numpy(bert_time_aligned).float()

        audio_wave, sr = librosa.load(os.path.join(data["audio_path"], data["video_id"] + ".wav"))
        audio_wave = librosa.resample(audio_wave, orig_sr=sr, target_sr=self.audio_sr)
        sdx_audio = sdx * int(1 / SMPLX_FPS * self.audio_sr)
        edx_audio = edx * int(1 / SMPLX_FPS * self.audio_sr)
        audio_wave = audio_wave[sdx_audio:edx_audio]
        audio_tensor = torch.from_numpy(audio_wave).float()

        return dict(
            cached_rep15d=motion_tensor,
            cached_audio_low=cached_audio_low,
            cached_audio_high=cached_audio_high,
            bert_time_aligned=bert_time_aligned,
            audio_tensor=audio_tensor,
        )