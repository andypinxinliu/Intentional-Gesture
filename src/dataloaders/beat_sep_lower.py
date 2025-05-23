import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pickle
import smplx
from .utils.audio_features import AudioProcessor
from .utils.other_tools import MultiLMDBManager
from .utils.motion_rep_transfer import process_smplx_motion
from .utils.mis_features import process_semantic_data, process_emotion_data
from .utils.text_features import process_word_data
from .utils.data_sample import sample_from_clip
from .utils import rotation_conversions as rc

class CustomDataset(Dataset):
    def __init__(self, args, loader_type, build_cache=True):
        self.args = args
        self.loader_type = loader_type
        self.rank = dist.get_rank()
        
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        
        # Initialize basic parameters
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0,0]  # for trinity
        
        # Initialize SMPLX model
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        
        self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
        
        # Load and process split rules
        self._process_split_rules()
        
        # Initialize data directories and lengths
        self._init_data_paths()
        
        # Build or load cache
        self._init_cache(build_cache)
        
    
    def _process_split_rules(self):
        """Process dataset split rules."""
        split_rule = pd.read_csv(self.args.data_path+"train_test_split.csv")
        self.selected_file = split_rule.loc[
            (split_rule['type'] == self.loader_type) & 
            (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
        ]
        
        if self.args.additional_data and self.loader_type == 'train':
            split_b = split_rule.loc[
                (split_rule['type'] == 'additional') & 
                (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
            ]
            self.selected_file = pd.concat([self.selected_file, split_b])
            
        if self.selected_file.empty:
            logger.warning(f"{self.loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[
                (split_rule['type'] == 'train') & 
                (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
            ]
            self.selected_file = self.selected_file.iloc[0:8]
    
    def _init_data_paths(self):
        """Initialize data directories and lengths."""
        self.data_dir = self.args.data_path
        
        if self.loader_type == "test":
            self.args.multi_length_training = [1.0]
            
        self.max_length = int(self.args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(self.args.pose_length / self.args.pose_fps * self.args.audio_sr)
        
        if self.max_audio_pre_len > self.args.test_length * self.args.audio_sr:
            self.max_audio_pre_len = self.args.test_length * self.args.audio_sr
        
        if self.args.test_clip and self.loader_type == "test":
            self.preloaded_dir = self.args.root_path + self.args.cache_path + self.loader_type + "_clip" + f"/{self.args.pose_rep}_cache"
        else:
            self.preloaded_dir = self.args.root_path + self.args.cache_path + self.loader_type + f"/{self.args.pose_rep}_cache"
    
    
    
    def _init_cache(self, build_cache):
        """Initialize or build cache."""
        self.lmdb_envs = {}
        self.mapping_data = None
        
        if build_cache and self.rank == 0:
            self.build_cache(self.preloaded_dir)
        
        self.load_db_mapping()
    
    def build_cache(self, preloaded_dir):
        """Build the dataset cache."""
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        
        if self.args.new_cache and os.path.exists(preloaded_dir):
            shutil.rmtree(preloaded_dir)
            
        if os.path.exists(preloaded_dir):
            # if the dir is empty, that means we still need to build the cache
            if not os.listdir(preloaded_dir):
                self.cache_generation(
                    preloaded_dir, 
                    self.args.disable_filtering,
                    self.args.clean_first_seconds,
                    self.args.clean_final_seconds,
                    is_test=False
                )
            else:
                logger.info("Found the cache {}".format(preloaded_dir))

        elif self.loader_type == "test":
            self.cache_generation(preloaded_dir, True, 0, 0, is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, 
                self.args.disable_filtering,
                self.args.clean_first_seconds,
                self.args.clean_final_seconds,
                is_test=False
            )
    
    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds, clean_final_seconds, is_test=False):
        """Generate cache for the dataset."""
        if not os.path.exists(out_lmdb_dir):
            os.makedirs(out_lmdb_dir)
        
        self.audio_processor = AudioProcessor(layer=self.args.n_layer, use_distill=self.args.use_distill)
        
        # Initialize the multi-LMDB manager
        lmdb_manager = MultiLMDBManager(out_lmdb_dir, max_db_size=10*1024*1024*1024)
        
        self.n_out_samples = 0
        n_filtered_out = defaultdict(int)
        
        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = os.path.join(self.data_dir, self.args.pose_rep, f_name + ext)
            
            # Process data
            data = self._process_file_data(f_name, pose_file, ext)
            if data is None:
                continue
                
            # Sample from clip
            filtered_result, self.n_out_samples = sample_from_clip(
                lmdb_manager=lmdb_manager,
                audio_file=pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav"),
                audio_each_file=data['audio_tensor'],
                high_each_file=data['high_level'],
                low_each_file=data['low_level'],
                pose_each_file=data['pose'],
                rep15d_each_file=data['rep15d'],
                trans_each_file=data['trans'],
                trans_v_each_file=data['trans_v'],
                shape_each_file=data['shape'],
                facial_each_file=data['facial'],
                aligned_text_each_file=data['aligned_text'],
                word_each_file=data['word'] if self.args.word_rep is not None else None,
                vid_each_file=data['vid'],
                emo_each_file=data['emo'],
                sem_each_file=data['sem'],
                intention_each_file=data['intention'] if data['intention'] is not None else None,
                audio_onset_each_file=data['audio_onset'] if self.args.onset_rep else None,
                args=self.args,
                ori_stride=self.ori_stride,
                ori_length=self.ori_length,
                disable_filtering=disable_filtering,
                clean_first_seconds=clean_first_seconds,
                clean_final_seconds=clean_final_seconds,
                is_test=is_test,
                n_out_samples=self.n_out_samples
            )
            
            for type_key in filtered_result:
                n_filtered_out[type_key] += filtered_result[type_key]
        
        lmdb_manager.close()
    
    def _process_file_data(self, f_name, pose_file, ext):
        """Process all data for a single file."""
        data = {
            'pose': None, 'trans': None, 'trans_v': None, 'shape': None,
            'audio': None, 'facial': None, 'word': None, 'emo': None,
            'sem': None, 'vid': None
        }
        
        # Process motion data
        logger.info(colored(f"# ---- Building cache for Pose {f_name} ---- #", "blue"))
        if "smplx" in self.args.pose_rep:
            motion_data = process_smplx_motion(pose_file, self.smplx, self.args.pose_fps, self.args.facial_rep)
        else:
            raise ValueError(f"Unknown pose representation '{self.args.pose_rep}'.")
            
        if motion_data is None:
            return None
            
        data.update(motion_data)
        
        # Process speaker ID
        if self.args.id_rep is not None:
            speaker_id = int(f_name.split("_")[0]) - 1
            data['vid'] = np.repeat(np.array(speaker_id).reshape(1, 1), data['pose'].shape[0], axis=0)
        else:
            data['vid'] = np.array([-1])
        
        # Process audio if needed
        if self.args.audio_rep is not None:
            audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
            audio_data = self.audio_processor.get_wav2vec_from_16k_wav(audio_file, aligned_text=True)
            if audio_data is None:
                return None
            data.update(audio_data)
        
        if getattr(self.args, "onset_rep", False):
            audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
            onset_data = self.audio_processor.calculate_onset_amplitude(audio_file, data)
            if onset_data is None:
                return None
            data.update(onset_data)
        
        # Process emotion if needed
        if self.args.emo_rep is not None:
            data = process_emotion_data(f_name, data, self.args)
            if data is None:
                return None
        
        # Process word data if needed
        if self.args.word_rep is not None:
            word_file = f"{self.data_dir}{self.args.word_rep}/{f_name}.TextGrid"
            data = process_word_data(self.data_dir, word_file, self.args, data, f_name, self.selected_file)
            if data is None:
                return None
        
        # Process intention data if needed
        if self.args.intention_rep is not None:
            intention_file = f"{self.data_dir}{self.args.intention_rep}/{f_name}.json"
            data = self.audio_processor.process_intention_data(intention_file, data, self.args)
            if data is None:
                return None
        
        # Process semantic data if needed
        if self.args.sem_rep is not None:
            sem_file = f"{self.data_dir}{self.args.sem_rep}/{f_name}.txt"
            data = process_semantic_data(sem_file, self.args, data, f_name)
            if data is None:
                return None
        
        return data
        
    def load_db_mapping(self):
        """Load database mapping from file."""
        mapping_path = os.path.join(self.preloaded_dir, "sample_db_mapping.pkl")
        with open(mapping_path, 'rb') as f:
            self.mapping_data = pickle.load(f)
        
        
        # Update paths from test to test_clip if needed
        if self.loader_type == "test" and self.args.test_clip:
            updated_paths = []
            for path in self.mapping_data['db_paths']:
                updated_path = path.replace("test/", "test_clip/")
                updated_paths.append(updated_path)
            self.mapping_data['db_paths'] = updated_paths
        
            # Re-save the updated mapping_data to the same pickle file
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.mapping_data, f)
        
        self.n_samples = len(self.mapping_data['mapping'])
    
    def get_lmdb_env(self, db_idx):
        """Get LMDB environment for given database index."""
        if db_idx not in self.lmdb_envs:
            db_path = self.mapping_data['db_paths'][db_idx]
            self.lmdb_envs[db_idx] = lmdb.open(db_path, readonly=True, lock=False)
        return self.lmdb_envs[db_idx]
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        db_idx = self.mapping_data['mapping'][idx]
        lmdb_env = self.get_lmdb_env(db_idx)
        
        with lmdb_env.begin(write=False) as txn:
            key = "{:008d}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            
            
            tar_pose, in_audio, in_audio_high, in_audio_low, tar_rep15d, in_facial, in_shape, in_aligned_text, in_word, emo, sem, vid, trans, trans_v, intention, audio_name, audio_onset = sample
            
            
            # Convert data to tensors with appropriate types
            processed_data = self._convert_to_tensors(
                tar_pose, tar_rep15d, in_audio, in_audio_high, in_audio_low, in_facial, in_shape, in_aligned_text, in_word,
                emo, sem, vid, trans, trans_v, intention, audio_onset
            )
            
            processed_data['audio_name'] = audio_name
            return processed_data
    
    def _convert_to_tensors(self, tar_pose, tar_rep15d, in_audio, in_audio_high, in_audio_low, in_facial, in_shape, in_aligned_text, in_word,
                           emo, sem, vid, trans, trans_v, intention=None, audio_onset=None):
        """Convert numpy arrays to tensors with appropriate types."""
        data = {
            'emo': torch.from_numpy(emo).int(),
            'sem': torch.from_numpy(sem).float(),
            'audio_tensor': torch.from_numpy(in_audio).float(),
            'audio_high': torch.from_numpy(in_audio_high).float(),
            'audio_low': torch.from_numpy(in_audio_low).float(),
            'bert_time_aligned': torch.from_numpy(in_aligned_text).float()
        }
        tar_pose = torch.from_numpy(tar_pose).float()
        
        if self.loader_type == "test":
            data.update({
                'pose': tar_pose,
                'rep15d': torch.from_numpy(tar_rep15d).float(),
                'trans': torch.from_numpy(trans).float(),
                'trans_v': torch.from_numpy(trans_v).float(),
                'facial': torch.from_numpy(in_facial).float(),
                'id': torch.from_numpy(vid).float(),
                'beta': torch.from_numpy(in_shape).float()
            })
        else:
            data.update({
                'pose': tar_pose,
                'rep15d': torch.from_numpy(tar_rep15d).reshape((tar_rep15d.shape[0], -1)).float(),
                'trans': torch.from_numpy(trans).reshape((trans.shape[0], -1)).float(),
                'trans_v': torch.from_numpy(trans_v).reshape((trans_v.shape[0], -1)).float(),
                'facial': torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float(),
                'id': torch.from_numpy(vid).reshape((vid.shape[0], -1)).float(),
                'beta': torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
            })
        
        # Handle intention
        if intention is not None and isinstance(intention, list) and len(intention) > 0:
            # Stack all embeddings into a list of tensors
            embeddings = [torch.from_numpy(item["embeddings"]).float() for item in intention]
            # Optionally, also collect text and timing if needed
            texts = [item["text"] for item in intention]
            timings = [item["timing"] for item in intention]
            data['intention_embeddings'] = embeddings  # list of [seq_len, hidden_size] tensors
            data['intention_texts'] = texts
            data['intention_timings'] = timings
            embedding_lengths = [item["embeddings"].shape[0] for item in intention]
            data['intention_embedding_lengths'] = embedding_lengths
        else:
            # No intention for this sample
            data['intention_embeddings'] = []
            data['intention_texts'] = []
            data['intention_timings'] = []
            data['intention_embedding_lengths'] = []
        
        # Handle audio onset
        if audio_onset is not None:
            data['audio_onset'] = torch.from_numpy(audio_onset).float()
        else:
            data['audio_onset'] = torch.tensor([-1])
        
        if in_word is not None:
            data['word'] = torch.from_numpy(in_word).int()
        else:
            data['word'] = torch.tensor([-1])
        
        return data