import math
import numpy as np
from collections import defaultdict
from loguru import logger

def sample_from_clip(
    lmdb_manager, audio_file, audio_each_file, high_each_file, low_each_file, pose_each_file, rep15d_each_file, trans_each_file, 
    trans_v_each_file, shape_each_file, facial_each_file, aligned_text_each_file, word_each_file,
    vid_each_file, emo_each_file, sem_each_file, intention_each_file, audio_onset_each_file, args, ori_stride, ori_length,
    disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
    n_out_samples):
    """Sample clips from the data according to specified parameters."""
    
    round_seconds_skeleton = pose_each_file.shape[0] // args.pose_fps

    # Calculate timing information
    timing_info = calculate_timing_info(
        audio_each_file, high_each_file, facial_each_file, round_seconds_skeleton, 
        args.audio_fps, args.pose_fps, args.audio_sr, args.audio_rep
    )
    
    round_seconds_skeleton = timing_info['final_seconds']
    
    # Calculate clip boundaries
    clip_info = calculate_clip_boundaries(
        round_seconds_skeleton, clean_first_seconds, clean_final_seconds,
        args.audio_fps, args.pose_fps
    )
    
    n_filtered_out = defaultdict(int)
    
    # Process each training length ratio
    for ratio in args.multi_length_training:
        processed_data = process_data_with_ratio(
            ori_stride, ori_length, ratio, clip_info, args, is_test, audio_each_file,
            high_each_file, low_each_file, pose_each_file, rep15d_each_file, trans_each_file, trans_v_each_file,
            shape_each_file, facial_each_file, aligned_text_each_file, word_each_file, vid_each_file,
            emo_each_file, sem_each_file, intention_each_file, audio_onset_each_file, audio_file,
            lmdb_manager, n_out_samples
        )
        
        for type_key, count in processed_data['filtered_counts'].items():
            n_filtered_out[type_key] += count
            
        n_out_samples = processed_data['n_out_samples']
    
    return n_filtered_out, n_out_samples

def calculate_timing_info(audio_data, audio_high_data, facial_data, round_seconds_skeleton, 
                         audio_fps, pose_fps, audio_sr, audio_rep):
    """Calculate timing information for the data."""
    if audio_high_data is not None:
        if audio_rep == 'wav2vec':
            round_seconds_audio_high = audio_high_data.shape[0] // 50
        elif audio_rep == "wave16k":
            round_seconds_audio_high = len(audio_high_data) // audio_fps
        elif audio_rep == "mfcc":
            round_seconds_audio_high = audio_high_data.shape[0] // audio_fps
        else:
            round_seconds_audio_high = audio_high_data.shape[0] // audio_sr
        
        round_seconds_audio = audio_data.shape[0] // audio_fps
        
        if facial_data is not None:
            round_seconds_facial = facial_data.shape[0] // pose_fps
            logger.info(f"audio: {round_seconds_audio}s, audio_high: {round_seconds_audio_high}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
            final_seconds = min(round_seconds_audio, round_seconds_audio_high, round_seconds_skeleton, round_seconds_facial)
            max_round = max(round_seconds_audio, round_seconds_audio_high, round_seconds_skeleton, round_seconds_facial)
            if final_seconds != max_round:
                logger.warning(f"reduce to {final_seconds}s, ignore {max_round-final_seconds}s")
        else:
            logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
            final_seconds = min(round_seconds_audio, round_seconds_skeleton)
            max_round = max(round_seconds_audio, round_seconds_skeleton)
            if final_seconds != max_round:
                logger.warning(f"reduce to {final_seconds}s, ignore {max_round-final_seconds}s")
    else:
        final_seconds = round_seconds_skeleton
        
    return {
        'final_seconds': final_seconds
    }

def calculate_clip_boundaries(round_seconds, clean_first_seconds, clean_final_seconds,
                            audio_fps, pose_fps):
    """Calculate the boundaries for clip sampling."""
    clip_s_t = clean_first_seconds
    clip_e_t = round_seconds - clean_final_seconds
    
    return {
        'clip_s_t': clip_s_t,
        'clip_e_t': clip_e_t,
        'clip_s_f_audio': audio_fps * clip_s_t,
        'clip_e_f_audio': clip_e_t * audio_fps,
        'clip_s_f_pose': clip_s_t * pose_fps,
        'clip_e_f_pose': clip_e_t * pose_fps
    }

def process_data_with_ratio(ori_stride, ori_length, ratio, clip_info, args, is_test, audio_data,
                           high_data, low_data, pose_data, rep15d_data, trans_data, trans_v_data,
                           shape_data, facial_data, aligned_text_data, word_data, vid_data,
                           emo_data, sem_data, intention_data, audio_onset_data, audio_file, 
                           lmdb_manager, n_out_samples):
    """Process data with a specific training length ratio."""
    
    if is_test and not args.test_clip:
        cut_length = clip_info['clip_e_f_pose'] - clip_info['clip_s_f_pose']
        args.stride = cut_length
        max_length = cut_length
    else:
        args.stride = int(ratio * ori_stride)
        cut_length = int(ori_length * ratio)
    
    
    num_subdivision = math.floor(
        (clip_info['clip_e_f_pose'] - clip_info['clip_s_f_pose'] - cut_length) / args.stride
    ) + 1
    
    logger.info(f"pose from frame {clip_info['clip_s_f_pose']} to {clip_info['clip_e_f_pose']}, length {cut_length}")
    logger.info(f"{num_subdivision} clips is expected with stride {args.stride}")
    
    if audio_data is not None:
        audio_short_length = math.floor(cut_length / args.pose_fps * args.audio_fps)
        logger.info(f"audio from frame {clip_info['clip_s_f_audio']} to {clip_info['clip_e_f_audio']}, length {audio_short_length}")
    
    if high_data is not None:
        high_short_length = math.floor(cut_length / args.pose_fps * 50)
        logger.info(f"high from frame {clip_info['clip_s_f_audio']} to {clip_info['clip_e_f_audio']}, length {high_short_length}")
    
    # Process subdivisions
    filtered_counts = defaultdict(int)
    for i in range(num_subdivision):
        sample_data = extract_sample_data(
            i, clip_info, cut_length, args, audio_data,
            high_data, low_data, pose_data, rep15d_data, trans_data, trans_v_data,
            shape_data, facial_data, aligned_text_data, word_data, vid_data,
            emo_data, sem_data, intention_data, audio_file, audio_onset_data,
            audio_short_length, high_short_length, is_test
        )
        
        if sample_data['pose'].any() is not None:
            lmdb_manager.add_sample([
                sample_data['pose'], sample_data['audio_tensor'], sample_data['high_level'], 
                sample_data['low_level'], sample_data['rep15d'],
                sample_data['facial'], sample_data['shape'], sample_data['aligned_text'], sample_data['word'], 
                sample_data['emo'], sample_data['sem'], sample_data['vid'], sample_data['trans'],
                sample_data['trans_v'], sample_data['intention'], sample_data['audio_name'], 
                sample_data['audio_onset']
            ])
            n_out_samples += 1
    
    return {
        'filtered_counts': filtered_counts,
        'n_out_samples': n_out_samples
    }

def extract_sample_data(idx, clip_info, cut_length, args, audio_data, 
                       high_data, low_data, pose_data, rep15d_data, trans_data, trans_v_data,
                       shape_data, facial_data, aligned_text_data, word_data, vid_data,
                       emo_data, sem_data, intention_data, audio_file, audio_onset_data,
                       audio_short_length, high_short_length, is_test):
    """Extract a single sample from the data."""
    start_idx = clip_info['clip_s_f_pose'] + idx * args.stride
    fin_idx = start_idx + cut_length
    
    # Convert frame indices to time
    segment_start_time = start_idx / args.pose_fps
    segment_end_time = fin_idx / args.pose_fps
    
    # Process intention data
    if intention_data is not None and args.intention_rep is not None:
        if not args.test_clip and is_test:
            # For test data without test_clip, include timing information
            intention_sequences = []
            for seq in intention_data:
                intention_sequences.append({
                    "timing": seq["timing"],
                    "embeddings": seq["embeddings"],
                    "text": seq["text"]
                })
            sample_intention = intention_sequences
        else:
            # For training data and test data with test_clip
            # Find intentions that overlap with current segment by at least 70%
            intention_sequences = []
            segment_duration = segment_end_time - segment_start_time
            
            for seq in intention_data:
                seq_start = seq["timing"]["start_time"]
                seq_end = seq["timing"]["end_time"]
                
                # Calculate overlap
                overlap_start = max(segment_start_time, seq_start)
                overlap_end = min(segment_end_time, seq_end)
                
                if overlap_end > overlap_start:  # There is overlap
                    overlap_duration = overlap_end - overlap_start
                    seq_duration = seq_end - seq_start
                    
                    # Check if overlap is at least 70% of the sequence duration
                    if overlap_duration >= 0.7 * seq_duration:
                        intention_sequences.append({
                            "embeddings": seq["embeddings"],
                            "text": seq["text"],
                            "timing": seq["timing"]
                        })
            
            sample_intention = intention_sequences if intention_sequences else None
    else:
        sample_intention = None
    
    sample_data = {
        'rep15d': rep15d_data[start_idx:fin_idx],
        'pose': pose_data[start_idx:fin_idx],
        'trans': trans_data[start_idx:fin_idx],
        'trans_v': trans_v_data[start_idx:fin_idx],
        'shape': shape_data[start_idx:fin_idx],
        'facial': facial_data[start_idx:fin_idx] if args.facial_rep is not None else np.array([-1]),
        'emo': emo_data[start_idx:fin_idx] if args.emo_rep is not None else np.array([-1]),
        'sem': sem_data[start_idx:fin_idx] if args.sem_rep is not None else np.array([-1]),
        'vid': vid_data[start_idx:fin_idx] if args.id_rep is not None else np.array([-1]),
        'intention': sample_intention if args.intention_rep is not None else np.array([-1]),
        'audio_name': audio_file
    }
    
    if audio_data is not None:
        audio_start = clip_info['clip_s_f_audio'] + math.floor(idx * args.stride * args.audio_fps / args.pose_fps)
        audio_end = audio_start + audio_short_length
        sample_data['audio_tensor'] = audio_data[audio_start:audio_end]
    else:
        sample_data['audio_tensor'] = np.array([-1])
    
    if high_data is not None:
        high_start = clip_info['clip_s_f_audio'] + math.floor(idx * args.stride * 50 / args.pose_fps)
        high_end = high_start + high_short_length
        sample_data['high_level'] = high_data[high_start:high_end]
        sample_data['low_level'] = low_data[high_start:high_end]
        sample_data['aligned_text'] = aligned_text_data[high_start:high_end]
    else:
        sample_data['high_level'] = np.array([-1])
        sample_data['low_level'] = np.array([-1])
        sample_data['aligned_text'] = np.array([-1])
    
    if audio_onset_data is not None:
        audio_onset_start = clip_info['clip_s_f_audio'] + math.floor(idx * args.stride * args.audio_fps / args.pose_fps)
        audio_onset_end = audio_onset_start + audio_short_length
        sample_data['audio_onset'] = audio_onset_data[audio_onset_start:audio_onset_end]
    else:
        sample_data['audio_onset'] = np.array([-1])
    
    if word_data is not None:
        sample_data['word'] = word_data[start_idx:fin_idx]
    else:
        sample_data['word'] = np.array([-1])
        
    return sample_data