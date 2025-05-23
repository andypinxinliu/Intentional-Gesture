import os
import shutil
import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import json 
import librosa
from datetime import datetime

import importlib
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

import wandb
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import smplx
from models.losses import InfoNCELoss, LocalContrastiveLoss




def train_val_fn(batch, model, device, mode="train", optimizer=None, lr_scheduler=None, mean_pose=None, std_pose=None):
    if mode == "train":
        model.train()
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
    else:
        model.eval()
        torch.set_grad_enabled(False)

    cached_rep15d = batch["rep15d"].to(device)
    cached_rep15d = (cached_rep15d - mean_pose) / std_pose
    cached_audio_low = batch["audio_low"].to(device) # (bs, T, C) C = 512
    cached_audio_high = batch["audio_high"].to(device) # (bs, T, C) C = 768
    bert_time_aligned = batch["bert_time_aligned"].to(device) # (bs, T, C) C = 768

    cached_audio_high = torch.cat([cached_audio_high, bert_time_aligned], dim=-1) # [bs, T, C] C = 1536 (768 + 768)
    
    audio_tensor = batch["audio_tensor"].to(device) # (bs, T) T = 68266
    
    intention_embeddings = batch.get("intention_embeddings", None)
    intention_texts = batch.get("intention_texts", None)
    intention_timings = batch.get("intention_timings", None)
    intention_mask = batch.get("intention_mask", None)
    
    if intention_embeddings is not None:
        intention_embeddings = intention_embeddings.to(device)
    if intention_mask is not None:
        intention_mask = intention_mask.to(device)
    
    model_out = model(cached_rep15d=cached_rep15d, cached_audio_low=cached_audio_low, cached_audio_high=cached_audio_high, in_audio=audio_tensor,
                     cached_intention=intention_embeddings, cached_intention_mask=intention_mask)
    audio_lower = model_out["audio_low"]
    motion_lower = model_out["motion_low"]
    audio_hihger_cls = model_out["audio_cls"]
    motion_higher_cls = model_out["motion_cls"]

    high_loss = model_out["high_level_loss"]
    low_infonce, low_acc = model_out["low_level_loss"]
    loss_dict = {
        "low_cosine": low_infonce,
        "high_infonce": high_loss
    }
    loss = sum(loss_dict.values())
    loss_dict["loss"] = loss
    loss_dict["low_acc"] = low_acc
    loss_dict["acc"] = compute_average_precision(audio_hihger_cls, motion_higher_cls)

    if mode == "train":    
        loss.backward()
        # clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        lr_scheduler.step()
    return loss_dict



def compute_average_precision(feature1, feature2):
    # Normalize the features
    feature1 = F.normalize(feature1, dim=1)
    feature2 = F.normalize(feature2, dim=1)
    
    # Compute the similarity matrix
    similarity_matrix = torch.matmul(feature1, feature2.t())
    
    # Get the top-1 predicted indices for each feature in feature1
    top1_indices = torch.argmax(similarity_matrix, dim=1)
    
    # Generate ground truth labels (diagonal indices)
    batch_size = feature1.size(0)
    ground_truth = torch.arange(batch_size, device=feature1.device)
    
    # Compute the accuracy (True if the top-1 index matches the ground truth)
    correct_predictions = (top1_indices == ground_truth).float()
    
    # Compute average precision
    average_precision = correct_predictions.mean()
    
    return average_precision
      

def evaluate_retrieval(model, test_loader, device, mean_pose, std_pose):
    model.eval()
    
    # Per-batch metrics (fixed size 128)
    batch_cls_correct_1 = 0  # Changed to accumulate raw counts
    batch_cls_correct_5 = 0
    batch_cls_correct_10 = 0
    batch_low_correct_1 = 0
    batch_low_correct_5 = 0
    batch_low_correct_10 = 0
    total_batch_samples = 0
    total_low_queries = 0
    num_full_batches = 0
    
    # Global collection
    all_audio_cls = []
    all_motion_cls = []
    all_audio_low = []
    all_motion_low = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Process batch
            cached_rep15d = batch["rep15d"].to(device)
            batch_size = cached_rep15d.size(0)
            
            # Skip non-128 batches for per-batch metrics
            if batch_size != 128:
                continue
                
            cached_rep15d = (cached_rep15d - mean_pose) / std_pose
            cached_audio_low = batch["audio_low"].to(device)
            cached_audio_high = batch["audio_high"].to(device)
            bert_time_aligned = batch["bert_time_aligned"].to(device)
            cached_audio_high = torch.cat([cached_audio_high, bert_time_aligned], dim=-1)
            audio_tensor = batch["audio_tensor"].to(device)
            
            # Get model outputs
            model_out = model(cached_rep15d=cached_rep15d, 
                            cached_audio_low=cached_audio_low, 
                            cached_audio_high=cached_audio_high, 
                            in_audio=audio_tensor)
            
            # High-level (CLS) features
            audio_cls = F.normalize(model_out["audio_cls"], dim=1)
            motion_cls = F.normalize(model_out["motion_cls"], dim=1)
            
            # Low-level features
            audio_low = F.normalize(model_out["audio_low"], dim=2)  # [B, T, D]
            motion_low = F.normalize(model_out["motion_low"], dim=2)  # [B, T, D]
            
            # Compute per-batch CLS metrics
            cls_similarity = torch.matmul(audio_cls, motion_cls.t())
            _, rankings = cls_similarity.topk(k=10, dim=1)
            
            # Count correct predictions for CLS
            for i in range(batch_size):
                target_idx = i
                if target_idx in rankings[i, :1]:
                    batch_cls_correct_1 += 1
                if target_idx in rankings[i, :5]:
                    batch_cls_correct_5 += 1
                if target_idx in rankings[i, :10]:
                    batch_cls_correct_10 += 1
            
            total_batch_samples += batch_size
            
            # Compute low-level recall@k for batch
            T = audio_low.size(1)
            
            # Motion to audio direction
            for t in range(T):
                motion_features = motion_low[:, t, :]  # [bs, D]
                audio_features = audio_low[:, t, :]    # [bs, D]
                
                # Compute similarities for all pairs within the batch
                similarities = torch.matmul(motion_features, audio_features.t())  # [bs, bs]
                
                # Get top-k indices
                _, top_indices = torch.topk(similarities, k=10, dim=1)  # [bs, k]
                
                # For each query in the batch
                for i in range(batch_size):
                    # Define positive window (±4 frames around the correct match)
                    correct_idx = i
                    start_idx = max(0, correct_idx - 4)
                    end_idx = min(batch_size, correct_idx + 4)
                    
                    # Check if top predictions fall within the positive window
                    correct_1 = ((top_indices[i, :1] >= start_idx) & (top_indices[i, :1] < end_idx)).any().item()
                    correct_5 = ((top_indices[i, :5] >= start_idx) & (top_indices[i, :5] < end_idx)).any().item()
                    correct_10 = ((top_indices[i, :10] >= start_idx) & (top_indices[i, :10] < end_idx)).any().item()
                    
                    batch_low_correct_1 += correct_1
                    batch_low_correct_5 += correct_5
                    batch_low_correct_10 += correct_10
                    total_low_queries += 1  # Increment for each query
            
            # Audio to motion direction (same process)
            for t in range(T):
                audio_features = audio_low[:, t, :]    # [bs, D]
                motion_features = motion_low[:, t, :]  # [bs, D]
                
                similarities = torch.matmul(audio_features, motion_features.t())  # [bs, bs]
                _, top_indices = torch.topk(similarities, k=10, dim=1)
                
                for i in range(batch_size):
                    correct_idx = i
                    start_idx = max(0, correct_idx - 4)
                    end_idx = min(batch_size, correct_idx + 4)
                    
                    correct_1 = ((top_indices[i, :1] >= start_idx) & (top_indices[i, :1] < end_idx)).any().item()
                    correct_5 = ((top_indices[i, :5] >= start_idx) & (top_indices[i, :5] < end_idx)).any().item()
                    correct_10 = ((top_indices[i, :10] >= start_idx) & (top_indices[i, :10] < end_idx)).any().item()
                    
                    batch_low_correct_1 += correct_1
                    batch_low_correct_5 += correct_5
                    batch_low_correct_10 += correct_10
                    total_low_queries += 1  # Increment for each query
            
            num_full_batches += 1
            
            # Store for global metrics
            all_audio_cls.append(audio_cls)
            all_motion_cls.append(motion_cls)
            all_audio_low.append(audio_low)
            all_motion_low.append(motion_low)
    
    # Ensure we have queries before division
    if total_low_queries == 0:
        raise ValueError("No queries were processed. Check batch size and data loading.")
        
    # Calculate per-batch metrics
    batch_cls_r1 = (batch_cls_correct_1 / total_batch_samples) * 100
    batch_cls_r5 = (batch_cls_correct_5 / total_batch_samples) * 100
    batch_cls_r10 = (batch_cls_correct_10 / total_batch_samples) * 100
    
    batch_low_r1 = (batch_low_correct_1 / total_low_queries) * 100
    batch_low_r5 = (batch_low_correct_5 / total_low_queries) * 100
    batch_low_r10 = (batch_low_correct_10 / total_low_queries) * 100
    
    # Global CLS metrics
    all_audio_cls = torch.cat(all_audio_cls, dim=0)
    all_motion_cls = torch.cat(all_motion_cls, dim=0)
    global_similarity = torch.matmul(all_audio_cls, all_motion_cls.t())
    _, rankings = global_similarity.topk(k=10, dim=1)
    
    total_samples = all_audio_cls.size(0)
    global_cls_correct_1 = 0
    global_cls_correct_5 = 0
    global_cls_correct_10 = 0
    
    for i in range(total_samples):
        target_idx = i
        if target_idx in rankings[i, :1]:
            global_cls_correct_1 += 1
        if target_idx in rankings[i, :5]:
            global_cls_correct_5 += 1
        if target_idx in rankings[i, :10]:
            global_cls_correct_10 += 1
    
    global_cls_r1 = (global_cls_correct_1 / total_samples) * 100
    global_cls_r5 = (global_cls_correct_5 / total_samples) * 100
    global_cls_r10 = (global_cls_correct_10 / total_samples) * 100
    
    # Global low-level metrics
    print("\nComputing global low-level metrics...")
    all_audio_low = torch.cat(all_audio_low, dim=0)   # [N, T, D]
    all_motion_low = torch.cat(all_motion_low, dim=0) # [N, T, D]
    total_samples = all_audio_low.size(0)
    T = all_audio_low.size(1)
    global_low_correct_1 = 0
    global_low_correct_5 = 0
    global_low_correct_10 = 0
    
    chunk_size = 32
    for chunk_start in tqdm(range(0, total_samples, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_size_actual = chunk_end - chunk_start
        
        # Motion to audio direction
        for t in range(T):
            motion_chunk = all_motion_low[chunk_start:chunk_end, t, :]  # [chunk_size, D]
            audio_features = all_audio_low[:, t, :]    # [N, D]
            
            # Compute similarities with all samples
            similarities = torch.matmul(motion_chunk, audio_features.t())  # [chunk_size, N]
            
            # Get top-k indices
            _, top_indices = torch.topk(similarities, k=10, dim=1)  # [chunk_size, k]
            
            # For each query in the chunk
            for i in range(chunk_size_actual):
                query_idx = chunk_start + i
                start_idx = max(0, query_idx - 4)
                end_idx = min(total_samples, query_idx + 4)
                
                correct_1 = ((top_indices[i, :1] >= start_idx) & (top_indices[i, :1] < end_idx)).any().item()
                correct_5 = ((top_indices[i, :5] >= start_idx) & (top_indices[i, :5] < end_idx)).any().item()
                correct_10 = ((top_indices[i, :10] >= start_idx) & (top_indices[i, :10] < end_idx)).any().item()
                
                global_low_correct_1 += correct_1
                global_low_correct_5 += correct_5
                global_low_correct_10 += correct_10
        
        # Audio to motion direction (same process)
        for t in range(T):
            audio_chunk = all_audio_low[chunk_start:chunk_end, t, :]
            motion_features = all_motion_low[:, t, :]
            
            similarities = torch.matmul(audio_chunk, motion_features.t())
            _, top_indices = torch.topk(similarities, k=10, dim=1)
            
            for i in range(chunk_size_actual):
                query_idx = chunk_start + i
                start_idx = max(0, query_idx - 4)
                end_idx = min(total_samples, query_idx + 4)
                
                correct_1 = ((top_indices[i, :1] >= start_idx) & (top_indices[i, :1] < end_idx)).any().item()
                correct_5 = ((top_indices[i, :5] >= start_idx) & (top_indices[i, :5] < end_idx)).any().item()
                correct_10 = ((top_indices[i, :10] >= start_idx) & (top_indices[i, :10] < end_idx)).any().item()
                
                global_low_correct_1 += correct_1
                global_low_correct_5 += correct_5
                global_low_correct_10 += correct_10
    
    total_global_queries = 2 * T * total_samples
    global_low_r1 = (global_low_correct_1 / total_global_queries) * 100
    global_low_r5 = (global_low_correct_5 / total_global_queries) * 100
    global_low_r10 = (global_low_correct_10 / total_global_queries) * 100

    return {
        "Per-batch CLS R@1": batch_cls_r1,
        "Per-batch CLS R@5": batch_cls_r5,
        "Per-batch CLS R@10": batch_cls_r10,
        "Per-batch Low-level R@1": batch_low_r1,
        "Per-batch Low-level R@5": batch_low_r5,
        "Per-batch Low-level R@10": batch_low_r10,
        "Global CLS R@1": global_cls_r1,
        "Global CLS R@5": global_cls_r5,
        "Global CLS R@10": global_cls_r10,
        "Global Low-level R@1": global_low_r1,
        "Global Low-level R@5": global_low_r5,
        "Global Low-level R@10": global_low_r10,
        "Num full batches": num_full_batches,
        "Total samples": total_samples
    }

def compute_recall_k(similarity_matrix):
    """Helper function to compute recall@k metrics"""
    _, rankings = similarity_matrix.topk(k=10, dim=1)
    total_samples = similarity_matrix.size(0)
    
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    
    for i in range(total_samples):
        target_idx = i
        pred_indices = rankings[i]
        if target_idx in pred_indices[:1]:
            recall_1 += 1
        if target_idx in pred_indices[:5]:
            recall_5 += 1
        if target_idx in pred_indices[:10]:
            recall_10 += 1
    
    return (recall_1 / total_samples * 100,
            recall_5 / total_samples * 100,
            recall_10 / total_samples * 100)


def custom_collate(batch):
    batch_out = {}
    for key in batch[0]:
        if key == 'intention_embeddings':
            cat_intentions = []
            lengths = []
            hidden_size = None
            for sample in batch:
                # sample[key] is a list of [seq_len, hidden] tensors
                if len(sample[key]) == 0:
                    # If no intentions, use a zero tensor of shape [1, hidden]
                    # We'll infer hidden size from the first non-empty sample
                    if hidden_size is None:
                        # Find hidden size from any non-empty sample
                        for s in batch:
                            if len(s[key]) > 0:
                                hidden_size = s[key][0].shape[1]
                                break
                        if hidden_size is None:
                            hidden_size = 768  # fallback
                    cat_intentions.append(torch.zeros(1, hidden_size))
                    lengths.append(0)
                else:
                    concat = torch.cat(sample[key], dim=0)
                    cat_intentions.append(concat)
                    lengths.append(concat.shape[0])
                    if hidden_size is None:
                        hidden_size = concat.shape[1]
            # Find max length in the batch
            max_len = max(lengths) if lengths else 1
            # Pad all to the same length
            padded = pad_sequence(cat_intentions, batch_first=True)  # [batch, max_total_seq_len, hidden]
            # Create mask: 1 for valid, 0 for padded
            mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
            for i, l in enumerate(lengths):
                if l > 0:
                    mask[i, :l] = 1
            batch_out[key] = padded
            batch_out['intention_mask'] = mask
        elif key.startswith('intention_'):
            # For texts and timings, keep as list of lists
            batch_out[key] = [item[key] for item in batch]
        else:
            try:
                batch_out[key] = default_collate([item[key] for item in batch])
            except Exception:
                batch_out[key] = [item[key] for item in batch]
    return batch_out

def main(cfg, args):
    # Set up distributed training environment
    import socket
    
    # Set default address and port
    master_addr = '127.0.0.1'
    master_port = 16075
    
    # Function to check if a port is in use
    def is_port_in_use(port, host='127.0.0.1'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False  # Port is available
            except socket.error:
                return True   # Port is in use
    
    # Find an available port
    while is_port_in_use(master_port):
        print(f"Port {master_port} is already in use, trying next port...")
        master_port += 1
    
    print(f"Using address {master_addr} and port {master_port}")
    
    # Set environment variables with validated address and port
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    local_rank = 0
    world_size = 1
    torch.distributed.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(cfg.seed)

    experiment_ckpt_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    
    # Load mean and std poses
    mean_pose = torch.from_numpy(np.load('./mean_std/rep15d_mean.npy')).to(device)
    std_pose = torch.from_numpy(np.load('./mean_std/rep15d_std.npy')).to(device)

    smplx_model = smplx.create(
            "datasets/hub/smplx_models", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(device).eval()

    model = init_class(cfg.model.name_pyfile, cfg.model.class_name, cfg).cuda()
    for param in model.parameters():
        param.requires_grad = True
    
    
    model.smplx_model = smplx_model
    model.high_level_loss_fn = InfoNCELoss()
    model.low_level_loss_fn = LocalContrastiveLoss()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,)
    
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    
    test_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg.data, loader_type='test')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=128, sampler=test_sampler, drop_last=False, num_workers=4, collate_fn=custom_collate)

    if args.mode == 'test':
        if args.checkpoint is None:
            raise ValueError("Must provide checkpoint path for test mode")
        
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Run evaluation
        print("Running retrieval evaluation...")
        metrics = evaluate_retrieval(model, test_loader, device, mean_pose, std_pose)
        
        print("\nRetrieval Results:")
        print("\nPer-batch Metrics (batch size 128):")
        print(f"CLS R@1: {metrics['Per-batch CLS R@1']:.2f}%")
        print(f"CLS R@5: {metrics['Per-batch CLS R@5']:.2f}%")
        print(f"CLS R@10: {metrics['Per-batch CLS R@10']:.2f}%")
        print(f"Low-level R@1: {metrics['Per-batch Low-level R@1']:.2f}%")
        print(f"Low-level R@5: {metrics['Per-batch Low-level R@5']:.2f}%")
        print(f"Low-level R@10: {metrics['Per-batch Low-level R@10']:.2f}%")
        
        
        print("\nGlobal Metrics:")
        print(f"CLS R@1: {metrics['Global CLS R@1']:.2f}%")
        print(f"CLS R@5: {metrics['Global CLS R@5']:.2f}%")
        print(f"CLS R@10: {metrics['Global CLS R@10']:.2f}%")
        print(f"Low-level R@1: {metrics['Global Low-level R@1']:.2f}%")
        print(f"Low-level R@5: {metrics['Global Low-level R@5']:.2f}%")
        print(f"Low-level R@10: {metrics['Global Low-level R@10']:.2f}%")
        print(f"Total samples: {metrics['Total samples']}")
        
        # Save results
        results_path = os.path.join(os.path.dirname(args.checkpoint), "retrieval_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Checkpoint: {args.checkpoint}\n\n")
            f.write("Per-batch Metrics (batch size 128):\n")
            f.write(f"CLS R@1: {metrics['Per-batch CLS R@1']:.2f}%\n")
            f.write(f"CLS R@5: {metrics['Per-batch CLS R@5']:.2f}%\n")
            f.write(f"CLS R@10: {metrics['Per-batch CLS R@10']:.2f}%\n")
            f.write(f"Low-level R@1: {metrics['Per-batch Low-level R@1']:.2f}%\n")
            f.write(f"Low-level R@5: {metrics['Per-batch Low-level R@5']:.2f}%\n")
            f.write(f"Low-level R@10: {metrics['Per-batch Low-level R@10']:.2f}%\n\n")
            f.write("Global Metrics:\n")
            f.write(f"CLS R@1: {metrics['Global CLS R@1']:.2f}%\n")
            f.write(f"CLS R@5: {metrics['Global CLS R@5']:.2f}%\n")
            f.write(f"CLS R@10: {metrics['Global CLS R@10']:.2f}%\n")
            f.write(f"Low-level R@1: {metrics['Global Low-level R@1']:.2f}%\n")
            f.write(f"Low-level R@5: {metrics['Global Low-level R@5']:.2f}%\n")
            f.write(f"Low-level R@10: {metrics['Global Low-level R@10']:.2f}%\n")
        
        print(f"\nResults saved to {results_path}")
    
    else:  # args.mode == 'train'
        # Initialize training-specific components
        train_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg.data, loader_type='train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.train_bs,
            sampler=train_sampler,
            drop_last=True,
            num_workers=4,
            collate_fn=custom_collate
        )
        
        # Resume from checkpoint if specified
        start_iteration = 0
        if args.checkpoint:
            print(f"Resuming from checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            start_iteration = checkpoint.get('iteration', 0) + 1

        if local_rank == 0:
            run_time = datetime.now().strftime("%Y%m%d-%H%M")
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.exp_name + "_" + run_time,
                entity=cfg.wandb_entity,
                dir=cfg.wandb_log_dir,
                config=OmegaConf.to_container(cfg)  # Pass config directly during initialization
            )
        else:
            writer = None
        
        num_epochs = cfg.solver.max_train_steps // len(train_loader) + 1
        iteration = start_iteration
        val_best = {}
        test_best = {}
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)

            for i, batch in enumerate(train_loader):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                loss_dict = train_val_fn(
                    batch, model, device, mode="train", optimizer=optimizer, lr_scheduler=lr_scheduler, mean_pose=mean_pose, std_pose=std_pose
                )
                if local_rank == 0 and iteration % cfg.log_period == 0:
                    for key, value in loss_dict.items():
                        # writer.add_scalar(f"train/{key}", value, iteration)
                        wandb.log({f"train/{key}": value}, step=iteration)
                    loss_message = ", ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])
                    print(f"Epoch {epoch} [{i}/{len(train_loader)}] - {loss_message}")

                if local_rank == 0 and iteration % cfg.validation.val_loss_steps == 0:
                    val_loss_dict = {}
                    val_batches = 0
                    for batch in tqdm(test_loader):
                        for k in batch:
                            if isinstance(batch[k], torch.Tensor):
                                batch[k] = batch[k].to(device)
                        loss_dict = train_val_fn(
                            batch, model, device, mode="val", optimizer=optimizer, lr_scheduler=lr_scheduler, mean_pose=mean_pose, std_pose=std_pose
                        )
                        for k, v in loss_dict.items():
                            if k not in val_loss_dict:
                                val_loss_dict[k] = 0
                            val_loss_dict[k] += v.item()  # Convert to float for accumulation
                        val_batches += 1
                        # if val_batches == 10:
                        #     break
                    val_loss_mean_dict = {k: v / val_batches for k, v in val_loss_dict.items()}
                    for k, v in val_loss_mean_dict.items():
                        # Initialize val_best with appropriate initial values if not exists
                        if k not in val_best:
                            if k in ["acc", "low_acc"]:
                                # For accuracy metrics, higher is better
                                val_best[k] = {"value": 0.0, "iteration": iteration}
                            else:
                                # For loss metrics (low_cosine, high_infonce), lower is better
                                val_best[k] = {"value": float('inf'), "iteration": iteration}
                        
                        # Update best values based on metric type
                        is_better = False
                        if k in ["acc", "low_acc"]:
                            # Higher is better for accuracy metrics
                            is_better = v > val_best[k]["value"]
                        else:
                            # Lower is better for loss metrics
                            is_better = v < val_best[k]["value"]
                        
                        if is_better:
                            val_best[k] = {"value": v, "iteration": iteration}
                            if k in ["acc", "low_acc"]:  # Save checkpoints for accuracy metrics
                                checkpoint_path = os.path.join(experiment_ckpt_dir, f"ckpt_{k}")
                                os.makedirs(checkpoint_path, exist_ok=True)
                                torch.save({
                                    'iteration': iteration,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                }, os.path.join(checkpoint_path, "ckpt.pth"))
                        
                        print(f"Val [{iteration}] - {k}: {v:.6f} (best: {val_best[k]['value']:.6f} at {val_best[k]['iteration']})")
                        wandb.log({f"val/{k}": v}, step=iteration)

                iteration += 1

        if local_rank == 0:
                wandb.finish()
    torch.distributed.destroy_process_group()

def init_class(module_name, class_name, config, **kwargs):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    instance = model_class(config, **kwargs)
    return instance


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def prepare_all():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/clip_baseline.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")
    parser.add_argument("--mode", type=str, choices=['train', 'test'], default='train',
                       help="Choose between 'train' or 'test' mode")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Checkpoint path for testing or resuming training")
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config.endswith(".yaml"):
        config = OmegaConf.load(args.config)
        config.exp_name = args.config.split("/")[-1][:-5]
    else:
        raise ValueError("Unsupported config file format. Only .yaml files are allowed.")

    if args.debug:
        config.wandb_project = "debug"
        config.exp_name = "debug"
        config.solver.max_train_steps = 4

    if args.overrides:
        for arg in args.overrides:
            key, value = arg.split('=')
            try:
                value = eval(value)
            except:
                pass
            if key in config:
                config[key] = value
            else:
                raise ValueError(f"Key {key} not found in config.")
    
    os.environ["WANDB_API_KEY"] = config.wandb_key

    save_dir = os.path.join(config.output_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)

    config_path = os.path.join(save_dir, 'sanity_check', f'{config.exp_name}.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sanity_check_dir = os.path.join(save_dir, 'sanity_check')
    
    # Get the absolute path of the output directory
    output_dir = os.path.abspath(config.output_dir)
    
    # Function to check if a path is within the output directory
    def is_in_output_dir(path):
        return os.path.abspath(path).startswith(output_dir)
    
    # Function to check if a file should be copied
    def should_copy_file(file_path):
        # Skip files in output directory
        if is_in_output_dir(file_path):
            return False
        # Skip __pycache__ directories
        if '__pycache__' in file_path:
            return False
        # Skip .pyc files
        if file_path.endswith('.pyc'):
            return False
        return True

    # Copy only Python files from the source directory
    for root, dirs, files in os.walk(current_dir):
        # Skip output directory and its subdirectories
        if is_in_output_dir(root):
            continue
            
        for file in files:
            if file.endswith(".py"):
                full_file_path = os.path.join(root, file)
                if should_copy_file(full_file_path):
                    relative_path = os.path.relpath(full_file_path, current_dir)
                    dest_path = os.path.join(sanity_check_dir, relative_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    try:
                        shutil.copy(full_file_path, dest_path)
                    except Exception as e:
                        print(f"Warning: Could not copy {full_file_path}: {str(e)}")
    
    return config, args


if __name__ == "__main__":
    config, args = prepare_all()
    main(config, args)