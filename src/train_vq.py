import os
import shutil
import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import json 
import librosa
from datetime import datetime
import math
import sys

import importlib
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from diffusers.optimization import get_scheduler
from dataloaders import data_tools
from tqdm import tqdm
from models.losses import hinge_loss, linear_loss, softplus_loss
from utils import other_tools
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate


def convert_15d_to_6d(motion):
    """
    Convert 15D motion to 6D motion, the current motion is 15D, but the eval model is 6D
    """
    bs = motion.shape[0]
    motion_6d = motion.reshape(bs, -1, 55, 15)[:, :, :, 6:12]
    motion_6d = motion_6d.reshape(bs, -1, 55*6)
    return motion_6d



def train_val_fn_simplified(batch, model, device, mode="train", optimizer=None, 
                 lr_scheduler=None, args=None, 
                 eval_copy=None, mean_pose=None, std_pose=None):
    """
    Simplified training and validation function without discriminator
    """
    if mode == "train":
        model.train()
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
    else:
        model.eval()
        torch.set_grad_enabled(False)

    # Process input data
    motion = batch["rep15d"].to(device)
    motion = (motion - mean_pose) / std_pose
    
    
    # Forward pass
    output = model(motion, ret_usages=True)
    
    rec_motion = output["motion_rec"]
    Lq, Le = output["vq_loss"], output["entropy_loss"]

    # Reconstruction loss
    L1 = F.l1_loss(rec_motion, motion)
    L2 = F.mse_loss(rec_motion, motion)
    # add 6d loss to avoid 15d loss dominate
    L2 += F.mse_loss(convert_15d_to_6d(rec_motion), convert_15d_to_6d(motion)) 
    Lrec = L1 * args.l1 + L2 * args.l2

    # Total loss (without discriminator)
    Lv = Lrec + args.lq * Lq + args.le * Le

    # Calculate FID and other metrics during validation
    fid = 0
    if eval_copy is not None:
        with torch.no_grad():
            gt_ori = motion.clone()
            n = motion.shape[1]
            
            # Apply normalization
            rec_motion_proc = rec_motion * std_pose + mean_pose
            gt_motion_proc = motion * std_pose + mean_pose
            
            # Handle sequence length
            remain = n % 32
            if remain != 0:
                rec_motion_proc = rec_motion_proc[:, :-remain]
                gt_motion_proc = gt_motion_proc[:, :-remain]
            
            # Get latent representations
            latent_rec = eval_copy.map2latent(convert_15d_to_6d(rec_motion_proc)).reshape(-1, 32)
            latent_gt = eval_copy.map2latent(convert_15d_to_6d(gt_motion_proc)).reshape(-1, 32)
            
            # Calculate FID
            fid = data_tools.FIDCalculator.frechet_distance(
                latent_rec.detach().cpu().numpy(),
                latent_gt.detach().cpu().numpy()
            )

    if mode == "train":
        Lv.backward()
        # Add gradient clipping for stability
        clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        lr_scheduler.step()
    
    loss_dict = {
        "L1": L1.item(),
        "L2": L2.item(),
        "Lrec": Lrec.item(),
        "Lq": Lq.item(),
        "Le": Le.item() if isinstance(Le, torch.Tensor) else float(Le),
        "codebook_usage": output["codebook_usages"] if "codebook_usages" in output else 0,
        "fid": float(fid),
    }
    
    return loss_dict

def evaluate_reconstruction(model, test_loader, device, mean_pose, std_pose, eval_copy=None):
    model.eval()
    torch.set_grad_enabled(False)
    
    # Initialize metrics
    l1_all = 0
    l2_all = 0
    fid_all = 0
    utility_ratio_all = 0
    total_batches = 0
    
    for batch in tqdm(test_loader, desc="Evaluating reconstruction"):
        # Process input data
        motion = batch["rep15d"].to(device)
        motion = (motion - mean_pose) / std_pose
        
        # Forward pass
        output = model(motion, ret_usages=True)
        rec_motion = output["motion_rec"]
        
        # Calculate L1 and L2 losses (before denormalization)
        l1 = F.l1_loss(rec_motion, motion).item()
        l2 = F.mse_loss(rec_motion, motion).item()
        
        # Calculate FID if eval_copy is provided
        fid = 0
        if eval_copy is not None:
            # Denormalize for FID calculation
            rec_motion_proc = rec_motion * std_pose + mean_pose
            gt_motion_proc = motion * std_pose + mean_pose
            
            # Handle sequence length
            n = motion.shape[1]
            remain = n % 32
            if remain != 0:
                rec_motion_proc = rec_motion_proc[:, :-remain]
                gt_motion_proc = gt_motion_proc[:, :-remain]
            
            # Get latent representations
            latent_rec = eval_copy.map2latent(convert_15d_to_6d(rec_motion_proc)).reshape(-1, 32)
            latent_gt = eval_copy.map2latent(convert_15d_to_6d(gt_motion_proc)).reshape(-1, 32)
            
            # Calculate FID
            fid = data_tools.FIDCalculator.frechet_distance(
                latent_rec.detach().cpu().numpy(),
                latent_gt.detach().cpu().numpy()
            )
        
        # Get codebook usage ratio
        utility_ratio = output["codebook_usages"] if "codebook_usages" in output else 0
        
        # Accumulate metrics
        l1_all += l1
        l2_all += l2
        fid_all += fid
        utility_ratio_all += utility_ratio
        total_batches += 1
    
    # Calculate averages
    metrics = {
        "L1": l1_all / total_batches,
        "L2": l2_all / total_batches,
        "FID": fid_all / total_batches,
        "Utility_Ratio": utility_ratio_all / total_batches
    }
    
    # Print results
    print("\nReconstruction Evaluation Results:")
    print(f"L1 Loss: {metrics['L1']:.6f}")
    print(f"L2 Loss: {metrics['L2']:.6f}")
    print(f"FID Score: {metrics['FID']:.6f}")
    print(f"Average Utility Ratio: {metrics['Utility_Ratio']:.6f}")
    
    return metrics


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
    
    
    # Add evaluation model setup
    eval_model_module = __import__(f"models.motion_representation", fromlist=["something"])
    eval_args = type('Args', (), {})()
    eval_args.vae_layer = 4
    eval_args.vae_length = 240
    eval_args.vae_test_dim = 330
    eval_args.variational = False
    eval_args.data_path_1 = "./datasets/hub/"
    eval_args.vae_grow = [1,1,2,1]
    
    eval_copy = getattr(eval_model_module, 'VAESKConv')(eval_args).to(device)
    other_tools.load_checkpoints(
        eval_copy, 
        './datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/weights/AESKConv_240_100.bin', 
        'VAESKConv'
    )
    
    # Load mean and std poses
    mean_pose = torch.from_numpy(np.load('./mean_std/rep15d_mean.npy')).to(device)
    std_pose = torch.from_numpy(np.load('./mean_std/rep15d_std.npy')).to(device)

    
    
    
    experiment_ckpt_dir = os.path.join(cfg.output_dir, cfg.exp_name)

    model = init_class(cfg.tokenizer.name_pyfile, cfg.tokenizer.class_name, cfg).cuda()
    for param in model.parameters():
        param.requires_grad = True
    
    discriminator = init_class(cfg.discriminator.name_pyfile, cfg.discriminator.class_name, cfg.discriminator).cuda()
    
    for param in discriminator.parameters():
        param.requires_grad = True
    

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon)

    # Initialize learning rate scheduler with num_cycles parameter
    lr_scheduler_kwargs = {
        "num_warmup_steps": cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        "num_training_steps": cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps,
    }

    # Add num_cycles parameter if it exists in the config
    if hasattr(cfg.solver, 'lr_scheduler_num_cycles'):
        lr_scheduler_kwargs["num_cycles"] = cfg.solver.lr_scheduler_num_cycles

    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        **lr_scheduler_kwargs
    )
    
    test_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg.data, loader_type='test')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=256, sampler=test_sampler, drop_last=False, num_workers=4, collate_fn=custom_collate)
    
    if args.mode == 'test':
        if args.checkpoint is None:
            raise ValueError("Must provide checkpoint path for test mode")
        
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # skp the pretrained encoder to prevent the legacy version error
        pretrained_state_dict = checkpoint['model_state_dict']
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'pretrained_encoder' not in k}
        model.load_state_dict(pretrained_state_dict, strict=False)
        model.eval()
        
        # Run evaluation
        print("Running reconstruction evaluation...")
        metrics = evaluate_reconstruction(model, test_loader, device, mean_pose, std_pose, eval_copy)
        
        
    

    
    else:
        train_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg.data, loader_type='train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=cfg.data.train_bs, sampler=train_sampler, drop_last=True, num_workers=4, collate_fn=custom_collate)
        # Resume from checkpoint if specified
        start_iteration = 0
        val_best = {}
        test_best = {}
        
        # Set up model warmup steps
        model_warmup_steps = 500  # The model warmup steps
        
        if hasattr(cfg, 'resume_from_checkpoint') and cfg.resume_from_checkpoint:
            print(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
            checkpoint = torch.load(os.path.join(cfg.resume_from_checkpoint, "ckpt.pth"), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            if 'iteration' in checkpoint:
                start_iteration = checkpoint['iteration'] + 1
                print(f"Resuming from iteration {start_iteration}")
            
            if 'val_best' in checkpoint:
                val_best = checkpoint['val_best']
                print(f"Resuming with best validation metrics: {val_best}")

        # Setup logging with wandb
        if local_rank == 0:
            run_time = datetime.now().strftime("%Y%m%d-%H%M")
            run_name = cfg.exp_name + "_" + run_time
            if start_iteration > 0:
                run_name += f"_resumed_{start_iteration}"
                
            wandb.init(
                project=cfg.wandb_project,
                name=run_name,
                entity=cfg.wandb_entity,
                dir=cfg.wandb_log_dir,
                config=OmegaConf.to_container(cfg)
            )
        
        # Calculate total epochs based on total steps and dataloader length
        num_epochs = (cfg.solver.max_train_steps - start_iteration) // len(train_loader) + 1
        iteration = start_iteration
        
        # Calculate starting epoch and batch index for efficient resuming
        start_epoch = start_iteration // len(train_loader)
        start_batch_idx = start_iteration % len(train_loader)
        
        print(f"Starting training for {num_epochs} epochs")
        if start_iteration > 0:
            print(f"Resuming from iteration {start_iteration} (epoch {start_epoch}, batch {start_batch_idx})")
        print(f"Model warmup for {model_warmup_steps} steps")
        
        
        # Main training loop
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_sampler.set_epoch(epoch)
            
            # Skip batches efficiently when resuming
            dataloader_iterator = iter(train_loader)
            if epoch == start_epoch and start_batch_idx > 0:
                print(f"Skipping to batch {start_batch_idx} of epoch {epoch}")
                # Skip to the starting batch for the first epoch when resuming
                for _ in range(start_batch_idx):
                    try:
                        next(dataloader_iterator)
                    except StopIteration:
                        break
            train_sampler.set_epoch(epoch)
            
            # Skip batches efficiently when resuming
            dataloader_iterator = iter(train_loader)
            if epoch == start_epoch and start_batch_idx > 0:
                print(f"Skipping to batch {start_batch_idx} of epoch {epoch}")
                # Skip to the starting batch for the first epoch when resuming
                for _ in range(start_batch_idx):
                    try:
                        next(dataloader_iterator)
                    except StopIteration:
                        break
            
            # Iterate through batches
            batch_idx = start_batch_idx if epoch == start_epoch else 0
            for i, batch in enumerate(dataloader_iterator, batch_idx):
                # Determine if we're in the model warmup phase (using lower learning rate)
                in_warmup = iteration < model_warmup_steps
                
                if in_warmup and local_rank == 0 and iteration % 10 == 0:
                    print(f"Model warmup phase: {iteration}/{model_warmup_steps}")
                
                # Training step using simplified function
                loss_dict = train_val_fn_simplified(
                    batch=batch,
                    model=model,
                    device=device,
                    mode="train",
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    args=cfg,
                    eval_copy=eval_copy,
                    mean_pose=mean_pose,
                    std_pose=std_pose
                )
                
                # Logging
                if local_rank == 0 and iteration % cfg.log_period == 0:
                    
                    for key, value in loss_dict.items():
                        wandb.log({f"train/{key}": value}, step=iteration)
                    
                    loss_message = ", ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])
                    status = "MODEL WARMUP" if in_warmup else "FULL TRAINING"
                    print(f"Epoch {epoch} [{i}/{len(train_loader)}] [{status}] - {loss_message}")

                # Validation
                if local_rank == 0 and iteration % cfg.validation.val_loss_steps == 0:
                    print(f"Running validation at iteration {iteration}")
                    val_loss_dict = {}
                    val_batches = 0
                    
                    for batch in tqdm(test_loader):
                        loss_dict = train_val_fn_simplified(
                            batch=batch, 
                            model=model,
                            device=device, 
                            mode="val", 
                            optimizer=optimizer, 
                            lr_scheduler=lr_scheduler,
                            args=cfg,
                            eval_copy=eval_copy,
                            mean_pose=mean_pose,
                            std_pose=std_pose
                        )
                        
                        for k, v in loss_dict.items():
                            if k not in val_loss_dict:
                                val_loss_dict[k] = 0
                            val_loss_dict[k] += v if isinstance(v, (int, float)) else v.item()
                        val_batches += 1
                        if val_batches == 10:
                            break
                            
                    val_loss_mean_dict = {k: v / val_batches for k, v in val_loss_dict.items()}
                    
                    for k, v in val_loss_mean_dict.items():
                        # Initialize val_best with appropriate initial values if not exists
                        if k not in val_best:
                            if k == "codebook_usage":
                                # For metrics where higher is better
                                val_best[k] = {"value": 0.0, "iteration": iteration}
                            elif k in ["fid", "L1", "L2", "Lrec", "Lq"]:
                                # For FID and loss metrics, initialize with a large value
                                val_best[k] = {"value": float('inf'), "iteration": iteration}
                            else:
                                # Skip metrics that don't need best tracking
                                continue
                        
                        # Update best values based on metric type
                        if k == "codebook_usage":
                            # Higher is better for codebook usage
                            if v > val_best[k]["value"]:
                                val_best[k] = {"value": v, "iteration": iteration}
                                checkpoint_path = os.path.join(experiment_ckpt_dir, f"ckpt_{k}")
                                os.makedirs(checkpoint_path, exist_ok=True)
                                torch.save({
                                    'iteration': iteration,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                    'val_best': val_best,
                                }, os.path.join(checkpoint_path, "ckpt.pth"))
                        elif k in ["fid", "L1", "L2", "Lrec", "Lq"]:
                            # Lower is better for these metrics
                            if v < val_best[k]["value"]:
                                val_best[k] = {"value": v, "iteration": iteration}
                                if k in ["fid", "L1", "L2", "Lrec"]:  # Save checkpoints for important metrics
                                    checkpoint_path = os.path.join(experiment_ckpt_dir, f"ckpt_{k}")
                                    os.makedirs(checkpoint_path, exist_ok=True)
                                    torch.save({
                                        'iteration': iteration,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                        'val_best': val_best,
                                    }, os.path.join(checkpoint_path, "ckpt.pth"))

                        if k in val_best:
                            print(f"Val [{iteration}] - {k}: {v:.6f} (best: {val_best[k]['value']:.6f} at {val_best[k]['iteration']})")
                        else:
                            print(f"Val [{iteration}] - {k}: {v:.6f}")
                        wandb.log({f"val/{k}": v}, step=iteration)
                    
                    # Save regular checkpoint
                    checkpoint_path = os.path.join(experiment_ckpt_dir, f"checkpoint_{iteration}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'val_best': val_best,
                    }, os.path.join(checkpoint_path, "ckpt.pth"))
                    
                    # Cleanup old checkpoints - keep only last 3
                    checkpoints = [d for d in os.listdir(experiment_ckpt_dir) if os.path.isdir(os.path.join(experiment_ckpt_dir, d)) and d.startswith("checkpoint_")]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1]))
                    if len(checkpoints) > 3:
                        for ckpt_to_delete in checkpoints[:-3]:
                            shutil.rmtree(os.path.join(experiment_ckpt_dir, ckpt_to_delete))

                iteration += 1
                if iteration >= cfg.solver.max_train_steps:
                    break
                    
            if iteration >= cfg.solver.max_train_steps:
                break

        # Final cleanup
        if local_rank == 0:
            wandb.finish()
        torch.distributed.destroy_process_group()


def init_class(module_name, class_name, config, **kwargs):
    """
    Dynamically import and initialize a class
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    instance = model_class(config, **kwargs)
    return instance


def seed_everything(seed):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_all():
    """
    Parse command line arguments and prepare configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/vq_baseline.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--model_warmup", type=int, default=500, help="Steps for model warmup before introducing discriminator")
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
    
    if args.resume:
        config.resume_from_checkpoint = args.resume
        
    if args.debug:
        config.wandb_project = "debug"
        config.exp_name = "debug"
        config.solver.max_train_steps = 4

    if args.overrides:
        for arg in args.overrides:
            if '=' in arg:
                key, value = arg.split('=')
                try:
                    value = eval(value)
                except:
                    pass
                if key in config:
                    config[key] = value
                else:
                    try:
                        # Handle nested config with dot notation
                        keys = key.split('.')
                        cfg_node = config
                        for k in keys[:-1]:
                            cfg_node = cfg_node[k]
                        cfg_node[keys[-1]] = value
                    except:
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