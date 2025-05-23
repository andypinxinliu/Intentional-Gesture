# semantic_utils.py
import pandas as pd
import numpy as np
from loguru import logger
import os

def process_semantic_data(sem_file, args, data, f_name):
    """Process semantic representation data."""
    logger.info(f"# ---- Building cache for Semantic {f_name} ---- #")
    
    if not os.path.exists(sem_file):
        logger.warning(f"# ---- file not found for Semantic {f_name} ---- #")
        return None
            
    sem_all = pd.read_csv(sem_file, 
        sep='\t', 
        names=["name", "start_time", "end_time", "duration", "score", "keywords"])
    
    sem_data = []
    for i in range(data['pose'].shape[0]):
        current_time = i/args.pose_fps
        found_score = False
        
        for _, row in sem_all.iterrows():
            if row['start_time'] <= current_time <= row['end_time']:
                sem_data.append(row['score'])
                found_score = True
                break
                
        if not found_score:
            sem_data.append(0.0)
    
    data['sem'] = np.array(sem_data)
    return data

def process_emotion_data(f_name, data, args):
    """Process emotion representation data."""
    logger.info(f"# ---- Building cache for Emotion {f_name} ---- #")
    
    rtype, start = int(f_name.split('_')[3]), int(f_name.split('_')[3])
    if rtype in [0, 2, 4, 6]:
        if 1 <= start <= 64:
            score = 0
        elif 65 <= start <= 72:
            score = 1
        elif 73 <= start <= 80:
            score = 2
        elif 81 <= start <= 86:
            score = 3
        elif 87 <= start <= 94:
            score = 4
        elif 95 <= start <= 102:
            score = 5
        elif 103 <= start <= 110:
            score = 6
        elif 111 <= start <= 118:
            score = 7
        else:
            score = 0
    else:
        score = 0
    
    data['emo'] = np.repeat(np.array(score).reshape(1, 1), data['pose'].shape[0], axis=0)
    return data

def process_intention_data(f_name, data, ):
    """Process intention representation data."""
    logger.info(f"# ---- Building cache for Intention {f_name} ---- #")
    
    pass