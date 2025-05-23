import numpy as np
import librosa
import math
import os
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import torch.nn.functional as F
import difflib
import soundfile as sf
from tqdm import tqdm
from transformers import (
    Wav2Vec2Model, 
    Wav2Vec2Config, 
    RobertaModel, 
    RobertaTokenizer,
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC
)
from models.wav2vec2.model import wav2vec2_model
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple
from numpy.lib import stride_tricks
from loguru import logger
import json
from models.layers.modality_encoder import MultiModalEncoder


class WrapedWav2Vec(nn.Module):
    def __init__(self, pretrained_model_name='facebook/wav2vec2-base-960h', layer=1):
        super(WrapedWav2Vec, self).__init__()
        base_model = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.feature_extractor = base_model.feature_extractor
        self.feature_projection = base_model.feature_projection
        self.encoder = base_model.encoder
        # Only use the three layer of the encoder
        self.encoder.layers = self.encoder.layers[:layer]
    
    def forward(self, 
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
        ):
        finetune_audio_low = self.feature_extractor(inputs).transpose(1, 2)
        hidden_states, _ = self.feature_projection(finetune_audio_low.detach())
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        
        return {
            "low_level": finetune_audio_low,
            "high_level": hidden_states
        }

class WarpedDistillWav2Vec(nn.Module):
    def __init__(self, pretrained_model_name='./datasets/DPWavLM-sp0.75.pth', layer=3):
        super(WarpedDistillWav2Vec, self).__init__()
        
        ckpt = torch.load(pretrained_model_name)
        wav2vec2_encoder = wav2vec2_model(**ckpt["config"])
        wav2vec2_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        
        self.feature_extractor = wav2vec2_encoder.feature_extractor
        self.feature_projection = wav2vec2_encoder.encoder.feature_projection
        self.encoder = wav2vec2_encoder.encoder.transformer
        self.encoder.layers = self.encoder.layers[:layer]
        

    def forward(self, 
        inputs,
        ):
        finetune_audio_low, _ = self.feature_extractor(inputs, length=None)
        hidden_states = self.feature_projection(finetune_audio_low.detach())
        encoder_outputs = self.encoder(
            hidden_states,
        )
        hidden_states = encoder_outputs
        
        return {
            "low_level": finetune_audio_low,
            "high_level": hidden_states
        }

class AudioProcessor:
    def __init__(self, device="cuda:0", layer=1, process_intention=False, use_distill=False):
        """
        Initialize the audio processor with all necessary models.
        
        Args:
            device (str): Device to run models on ('cuda:0', 'cpu', etc.)
        """
        self.device = device
        
        # Initialize all required models
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        if use_distill:
            self.wav2vec2_encoder = WarpedDistillWav2Vec(layer=layer).to(device)
        else:
            self.wav2vec2_encoder = WrapedWav2Vec(layer=layer).to(device)
        self.wav2vec2_encoder.eval()
        
        self.bert_model = RobertaModel.from_pretrained("roberta-base").to(device)
        self.bert_model.eval()
        self.bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        self.asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device)
        self.asr.eval()
        
        self.multi_modal_encoder = MultiModalEncoder(args)
        self.multi_modal_encoder.eval()
        self.multi_modal_encoder.load_from_pretrained(args.multi_modal_encoder_path)

    def audio_to_time_aligned_text_features(self, input_values):
        """
        Extract time-aligned text features from audio inputs.
        
        Args:
            input_values (torch.Tensor): Audio input values
            
        Returns:
            Tuple of transcription, features per timestep, and all token embeddings
        """
        with torch.no_grad():
            logits = self.asr(input_values).logits  # shape: (1, time_steps, vocab_size)

        predicted_ids_per_timestep = torch.argmax(logits, dim=-1)  # shape: (1, time_steps)
        predicted_ids_per_timestep = predicted_ids_per_timestep[0].cpu().numpy()
        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        tokens_per_timestep = [id_to_token[id] for id in predicted_ids_per_timestep]

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec2_processor.decode(predicted_ids[0])
        inputs_bert = self.bert_tokenizer(transcription, return_tensors='pt')
        input_ids = inputs_bert['input_ids'][0]  
        tokens_bert = self.bert_tokenizer.convert_ids_to_tokens(input_ids)

        with torch.no_grad():
            outputs_bert = self.bert_model(**inputs_bert.to(input_values.device))
        all_token_embeddings = outputs_bert.last_hidden_state[0]  
        
        per_timestep_chars = []
        per_timestep_char_indices = []
        for idx, t in enumerate(tokens_per_timestep):
            if t not in ('<pad>', '|'):
                per_timestep_chars.append(t.lower())
                per_timestep_char_indices.append(idx)
        
        bert_chars = []
        bert_char_indices = []
        for idx, token in enumerate(tokens_bert):
            if token in ('[CLS]', '[SEP]'):
                continue
            token_str = token.replace('##', '')
            for c in token_str:
                bert_chars.append(c)
                bert_char_indices.append(idx)

        s = difflib.SequenceMatcher(None, per_timestep_chars, bert_chars)
        opcodes = s.get_opcodes()
        per_timestep_to_bert_token_idx = {}
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                for k in range(i2 - i1):
                    per_timestep_idx = per_timestep_char_indices[i1 + k]
                    bert_token_idx = bert_char_indices[j1 + k]
                    per_timestep_to_bert_token_idx[per_timestep_idx] = bert_token_idx
        
        features_per_timestep = []
        check = []
        for i, per_token in enumerate(tokens_per_timestep):
            if i == 0:
                embedding = all_token_embeddings[0]
                check.append("cls")
            elif per_token in ('<pad>', '|'):
                embedding = torch.zeros(all_token_embeddings.shape[-1]).to(input_values.device)
                check.append(0)
            else:
                if i in per_timestep_to_bert_token_idx:
                    bert_idx = per_timestep_to_bert_token_idx[i]
                    embedding = all_token_embeddings[bert_idx]
                    check.append(tokens_bert[bert_idx])
                else:
                    embedding = torch.zeros(all_token_embeddings.shape[-1]).to(input_values.device)
                    check.append(0)
            features_per_timestep.append(embedding)
        
        features_per_timestep = torch.stack(features_per_timestep)  

        updated_check = check.copy()
        for i in range(len(check)):
            if check[i] == 0:
                left = i - 1
                right = i + 1
                left_found = False
                right_found = False

                while left >= 0:
                    if check[left] != 0:
                        left_found = True
                        break
                    left -= 1

                while right < len(check):
                    if check[right] != 0:
                        right_found = True
                        break
                    right += 1

                if left_found and right_found:
                    if (i - left) <= (right - i):
                        nearest = left
                    else:
                        nearest = right
                elif left_found:
                    nearest = left
                elif right_found:
                    nearest = right
                else:
                    continue
                updated_check[i] = updated_check[nearest]
                features_per_timestep[i] = features_per_timestep[nearest]
        
        features_per_timestep = features_per_timestep
        return transcription, features_per_timestep, all_token_embeddings

    def get_wav2vec_from_16k_wav(self, wav_16k_name, aligned_text=False):
        """
        Extract wav2vec features from a 16kHz wav file.
        
        Args:
            wav_16k_name (str): Path to the 16kHz wav file
            aligned_text (bool): Whether to extract aligned text features
            
        Returns:
            Dict containing high-level and low-level features
        """
        speech_16k, _ = sf.read(wav_16k_name)
        audio_feats = self.get_wav2vec_from_16k_speech(speech_16k, aligned_text)
        audio_feats["audio_tensor"] = speech_16k
        return audio_feats

    @torch.no_grad()
    def get_wav2vec_from_16k_speech(self, speech, aligned_text=False):
        """
        Extract wav2vec features from 16kHz speech array.
        
        Args:
            speech (np.array): Speech array
            aligned_text (bool): Whether to extract aligned text features
            
        Returns:
            Dict containing high-level and low-level features, and optionally aligned text
        """
        if speech.ndim == 2:
            speech = speech[:, 0]  # [T, 2] ==> [T,]
            
        input_values_all = self.wav2vec2_processor(
            speech, return_tensors="pt", sampling_rate=16000
        ).input_values  # [1, T]
        
        input_values_all = input_values_all.to(self.device)
        
        # For long audio sequence, due to the memory limitation, we cannot process them in one run
        # Wav2Vec process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
        # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
        # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
        # We have the equation to calculate out time step: T = floor((t-k)/s)
        # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
        # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        num_iter = input_values_all.shape[1] // clip_length
        expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
        res_lst_high = []
        res_lst_low = []

        if aligned_text:
            res_lst_text = []

        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
                
            input_values = input_values_all[:, start_idx: end_idx]
            hidden_states = self.wav2vec2_encoder(input_values)  # [B=1, T=pts//320, hid=1024]
            res_lst_high.append(hidden_states["high_level"][0])
            res_lst_low.append(hidden_states["low_level"][0])

            if aligned_text:
                _, features_per_timestep, _ = self.audio_to_time_aligned_text_features(
                    input_values
                )
                res_lst_text.append(features_per_timestep)

        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all
            
        # Process the last batch if it's long enough
        if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it            
            hidden_states = self.wav2vec2_encoder(input_values)  # [B=1, T=pts//320, hid=1024]
            res_lst_high.append(hidden_states["high_level"][0])
            res_lst_low.append(hidden_states["low_level"][0])

            if aligned_text:
                _, features_per_timestep, _ = self.audio_to_time_aligned_text_features(
                    input_values
                )
                res_lst_text.append(features_per_timestep)
            
        ret_high = torch.cat(res_lst_high, dim=0).cpu()  # [T, 1024]
        ret_low = torch.cat(res_lst_low, dim=0).cpu()  # [T, 256]
        
        if aligned_text:
            ret_text = torch.cat(res_lst_text, dim=0).cpu()

        # Verify the output shape is close to expected
        assert abs(ret_high.shape[0] - expected_T) == abs(ret_low.shape[0] - expected_T) <= 1

        if aligned_text:
            assert ret_high.shape[0] == ret_text.shape[0]

        # Adjust output to match expected length
        if ret_high.shape[0] < expected_T:
            ret_high = torch.nn.functional.pad(ret_high, (0, 0, 0, expected_T-ret_high.shape[0]))
            ret_low = torch.nn.functional.pad(ret_low, (0, 0, 0, expected_T-ret_low.shape[0]))

            if aligned_text:
                ret_text = torch.nn.functional.pad(ret_text, (0, 0, 0, expected_T-ret_text.shape[0]))
        else:
            ret_high = ret_high[:expected_T]
            ret_low = ret_low[:expected_T]

            if aligned_text:
                ret_text = ret_text[:expected_T]
        
        # # downsample， /5 * 3
        # ret_high = F.interpolate(ret_high.transpose(0,1).unsqueeze(0), scale_factor=0.6, mode='linear', align_corners=False).squeeze(0).transpose(0,1)
        # ret_low = F.interpolate(ret_low.transpose(0,1).unsqueeze(0), scale_factor=0.6, mode='linear', align_corners=False).squeeze(0).transpose(0,1)
        # if aligned_text:
        #     ret_text = F.interpolate(ret_text.transpose(0,1).unsqueeze(0), scale_factor=0.6, mode='linear', align_corners=False).squeeze(0).transpose(0,1)
        
        # convert to numpy
        ret_high = ret_high.numpy()
        ret_low = ret_low.numpy()
        if aligned_text:
            ret_text = ret_text.numpy()

        return_dict = {
            "high_level": ret_high,
            "low_level": ret_low,
        }

        if aligned_text:
            return_dict["aligned_text"] = ret_text

        return return_dict

    @staticmethod
    def extract_melspec(file, destpath, fps, n_mels=128):
        """
        Extract mel-spectrogram features from audio file.
        
        Args:
            file (str): Path to audio file
            destpath (str): Path to save the extracted features
            fps (int): Frames per second
            n_mels (int): Number of mel bands
        """
        fs, X = wav.read(file)
        X = X.astype(float) / math.pow(2, 15)
        target_sr = 48000
        X_48k = librosa.resample(X, orig_sr=fs, target_sr=target_sr, res_type="kaiser_best")
        n_fft = int(target_sr * 0.13)
        hop_len = int(target_sr / fps)
        C = librosa.feature.melspectrogram(
            y=X_48k, sr=target_sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels, fmin=0.0, fmax=8000
        )
        C = np.log(C)
        np.save(destpath, np.transpose(C))

    def process_audio_data(self, audio_file, args, data, f_name, selected_file):
        """
        Process audio data with support for different representations.
        
        Args:
            audio_file (str): Path to audio file
            args: Processing arguments
            data (dict): Data dictionary to store results
            f_name (str): File name identifier
            selected_file: DataFrame with file information
            
        Returns:
            Updated data dictionary or None if processing failed
        """
        logger.info(f"# ---- Building cache for Audio {f_name} ---- #")
        
        if not os.path.exists(audio_file):
            logger.warning(f"# ---- file not found for Audio {f_name}, skip all files with the same id ---- #")
            selected_file.drop(selected_file[selected_file['id'] == f_name].index, inplace=True)
            return None

        audio_save_path = audio_file.replace("wave16k", "onset_amplitude").replace(".wav", ".npy")
        
        if args.audio_rep == "onset+amplitude" and os.path.exists(audio_save_path):
            data['audio'] = np.load(audio_save_path)
            logger.warning(f"# ---- file found cache for Audio {f_name} ---- #")
        
        elif args.audio_rep == "onset+amplitude":
            data['audio'] = self.calculate_onset_amplitude(audio_file, args.audio_sr, audio_save_path)
            
        elif args.audio_rep == "mfcc":
            audio_data, _ = librosa.load(audio_file)
            data['audio'] = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=args.audio_sr, 
                n_mels=128, 
                hop_length=int(args.audio_sr/args.audio_fps)
            ).transpose(1, 0)
        
        if args.audio_norm and args.audio_rep == "wave16k":
            data['audio'] = (data['audio'] - args.mean_audio) / args.std_audio
        
        # wav2vec2 feature processing
        if args.audio_rep == "wav2vec2":
            audio_save_path = audio_file.replace("wave16k", f"wav2vec2_{args.audio_fps}").replace(".wav", ".npy")
            if os.path.exists(audio_save_path):
                data['audio'] = np.load(audio_save_path)
                logger.warning(f"# ---- file found cache for Audio {f_name} ---- #")
            else:
                audio_data, _ = librosa.load(audio_file)
                audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=args.audio_sr)
                audio_data = (audio_data - args.mean_audio) / args.std_audio
                
                # Using this class's method instead of global model
                audio_feat = self.get_wav2vec_from_16k_speech(audio_data)
                
                # Save the high-level features
                np.save(audio_save_path, audio_feat["high_level"].numpy())
                data['audio'] = audio_feat["high_level"].numpy()
        
        return data

    @staticmethod
    def calculate_onset_amplitude(audio_file, data, audio_sr=16000):
        """
        Calculate onset and amplitude features from audio file.
        
        Args:
            audio_file (str): Path to audio file
            save_path (str): Path to save the extracted features
            
        Returns:
            np.array: Extracted features
        """
        audio_save_path = audio_file.replace("wave16k", "onset_amplitude").replace(".wav", ".npy")
        if os.path.exists(audio_save_path):
            data['audio_onset'] = np.load(audio_save_path)
            logger.warning(f"# ---- file found cache for Audio {audio_file} ---- #")
            return data
        
        audio_data, sr = librosa.load(audio_file)
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=audio_sr)
        
        # Calculate amplitude envelope
        frame_length = 1024
        shape = (audio_data.shape[-1] - frame_length + 1, frame_length)
        strides = (audio_data.strides[-1], audio_data.strides[-1])
        rolling_view = stride_tricks.as_strided(audio_data, shape=shape, strides=strides)
        amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
        amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
        
        # Calculate onset
        audio_onset_f = librosa.onset.onset_detect(y=audio_data, sr=audio_sr, units='frames')
        onset_array = np.zeros(len(audio_data), dtype=float)
        onset_array[audio_onset_f] = 1.0
        
        # Combine features
        features = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
        
        # Save features
        os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)
        np.save(audio_save_path, features)
        
        data['audio_onset'] = features
        return data
    
    def process_intention_data(self, intention_file, data, args):
        """
        Process intention data from JSON files and extract intention embeddings.
        Each sequence will have its full token embeddings preserved.
        
        Args:
            intention_file (str): Path to the intention JSON file
            data (dict): Data dictionary to store results
            args: Processing arguments
            
        Returns:
            dict: Updated data dictionary with sequence-level intention embeddings
        """
        try:
            with open(intention_file, 'r') as f:
                intention_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading intention file {intention_file}: {str(e)}")
            return None

        # Get all sequences
        sequences = intention_data.get("sequences", [])
        if not sequences:
            logger.warning(f"No sequences found in {intention_file}")
            return None

        # Initialize list to store sequence data
        sequence_embeddings = []
        
        # Process each sequence
        for sequence in sequences:
            # Extract timing information
            timing = sequence.get("sequence_timing", {})
            start_time = timing.get("start_time", 0)
            end_time = timing.get("end_time", 0)
            
            # Extract intention summary
            analysis = sequence.get("analysis", "")
            summary = ""
            
            # Find the direct intention summary section
            if "Direct Intention Summary" in analysis:
                summary = analysis.split("Direct Intention Summary")[1].strip().replace("**", "").replace(":", "").replace("\n", "")
            
            if not summary:
                logger.warning(f"No intention summary found for sequence {start_time}-{end_time}")
                continue
            
            # Encode the summary using BERT
            with torch.no_grad():
                inputs = self.bert_tokenizer(summary, 
                                           return_tensors='pt',
                                           padding=True, 
                                           truncation=True, 
                                           max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.bert_model(**inputs)
                
                # Get all token embeddings (excluding special tokens)
                token_embeddings = outputs.last_hidden_state[0, 1:-1].cpu().numpy()  # Shape: [seq_len, hidden_size]
            
            # Store sequence data
            sequence_data = {
                "timing": {
                    "start_time": start_time,
                    "end_time": end_time
                },
                "embeddings": token_embeddings,  # Full sequence of token embeddings
                "text": summary  # Store original text for reference
            }
            
            sequence_embeddings.append(sequence_data)
        
        if not sequence_embeddings:
            logger.warning(f"No valid embeddings extracted from {intention_file}")
            return None
        
        # Store all sequence embeddings in the data dictionary
        data['intention'] = sequence_embeddings
        
        return data

# Example usage
if __name__ == "__main__":
    # Example initialization and usage
    processor = AudioProcessor(device="cuda:0")
    
    # Example of processing a 16kHz wav file
    # audio_features = processor.get_wav2vec_from_16k_wav("path/to/audio.wav")
    
    # Example of extracting mel-spectrogram
    # processor.extract_melspec("path/to/audio.wav", "path/to/save.npy", fps=15, n_mels=128)
    
    # Example for processing with arguments
    class Args:
        audio_rep = "wav2vec2"
        audio_sr = 16000
        audio_fps = 15
        audio_norm = True
        mean_audio = 0
        std_audio = 1
    
    args = Args()
    # processor.process_audio_data("path/to/audio.wav", args, {}, "file_id", selected_file_df)