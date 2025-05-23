import os
import pickle
import numpy as np
import torch
import lmdb
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from tqdm import tqdm


class StatsDataset(Dataset):
    """
    Dataset to calculate statistics (mean/std) of the motion representation data.
    Reuses core components from CustomDataset but only loads rep15d.
    """
    def __init__(self, args, loader_type="train"):
        self.args = args
        self.loader_type = loader_type
        
        # Initialize distributed rank
        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0
        
        # Set cache path
        self.preloaded_dir = os.path.join(
            self.args.root_path, 
            self.args.cache_path, 
            self.loader_type, 
            f"{self.args.pose_rep}_cache"
        )
        
        # Load database mapping
        self.load_db_mapping()
        self.lmdb_envs = {}
    
    def load_db_mapping(self):
        """Load database mapping from file."""
        mapping_path = os.path.join(self.preloaded_dir, "sample_db_mapping.pkl")
        with open(mapping_path, 'rb') as f:
            self.mapping_data = pickle.load(f)
        self.n_samples = len(self.mapping_data['mapping'])
        print(f"Loaded {self.n_samples} samples from {len(self.mapping_data['db_paths'])} databases")
    
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
        """Get only the rep15d representation from a sample."""
        db_idx = self.mapping_data['mapping'][idx]
        lmdb_env = self.get_lmdb_env(db_idx)
        
        with lmdb_env.begin(write=False) as txn:
            key = "{:008d}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            
            # Unpack only the rep15d (index 4 in the sample tuple)
            _, _, _, _, tar_rep15d, _, _, _, _, _, _, _, _, _ = sample
            
            # Return as tensor
            return torch.from_numpy(tar_rep15d).float()


def calculate_motion_statistics(args):
    """
    Calculate mean and standard deviation for the rep15d motion representation.
    
    Args:
        args: Configuration arguments including dataset paths
        
    Returns:
        mean: Mean of rep15d motion data (shape: 15*55)
        std: Standard deviation of rep15d motion data (shape: 15*55)
    """
    print("Loading dataset for statistics calculation...")
    
    # Create dataset instance
    dataset = StatsDataset(args, loader_type="train")
    
    # Use DataLoader with multiple workers for faster processing
    dataloader = DataLoader(
        dataset, 
        batch_size=64,  # Process multiple samples at once
        shuffle=False,  # No need to shuffle for statistics
        num_workers=8,  # Adjust based on your machine's capabilities
        pin_memory=True
    )
    
    # Initialize variables for statistics
    sum_over_batch = None
    sum_squared_over_batch = None
    total_sequences = 0
    feature_dim = None
    
    print("Computing statistics...")
    
    # First pass: calculate sum and sum of squares for mean and variance
    for batch in tqdm(dataloader):
        # batch shape: [batch_size, seq_len, 15*55]
        if feature_dim is None:
            feature_dim = batch.shape[2]
            sum_over_batch = torch.zeros(feature_dim)
            sum_squared_over_batch = torch.zeros(feature_dim)
        
        # For each sample, average across sequence length
        batch_means = batch.mean(dim=1)  # [batch_size, 15*55]
        batch_means_squared = (batch ** 2).mean(dim=1)  # [batch_size, 15*55]
        
        # Sum across batch
        sum_over_batch += batch_means.sum(dim=0)
        sum_squared_over_batch += batch_means_squared.sum(dim=0)
        
        # Track total number of sequences
        total_sequences += batch.shape[0]
    
    # Calculate mean and standard deviation
    mean = sum_over_batch / total_sequences
    
    # Modified approach for calculating variance to avoid numerical precision issues
    # We compute: var = E[(X - E[X])²] directly for better numerical stability
    sum_sq_diff = torch.zeros_like(sum_over_batch)
    
    print("Computing variance with stable algorithm...")
    
    # Second pass through the data to calculate variance stably
    for batch in tqdm(dataloader):
        # For each sample, calculate (X - mean)² directly, then average over sequence
        batch_centered = batch - mean.unsqueeze(0).unsqueeze(0)  # Broadcast mean to batch shape
        batch_centered_squared = batch_centered ** 2
        batch_var_contribution = batch_centered_squared.mean(dim=1)  # Average over sequence length
        
        # Accumulate
        sum_sq_diff += batch_var_contribution.sum(dim=0)
    
    # Variance is the average of squared differences
    variance = sum_sq_diff / total_sequences
    
    # Standard deviation with small epsilon for numerical stability
    std = torch.sqrt(variance + 1e-8)
    
    print(f"Processed {total_sequences} sequences")
    print(f"Feature dimension: {feature_dim}")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    
    # Save statistics
    output_dir = os.path.join(args.root_path, "statistics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    mean_np = mean.numpy()
    std_np = std.numpy()
    
    # Save as separate NumPy files
    np.save(os.path.join(output_dir, "rep15d_mean.npy"), mean_np)
    np.save(os.path.join(output_dir, "rep15d_std.npy"), std_np)
    
    # Also save as pickle for additional metadata
    stats_file = os.path.join(output_dir, "rep15d_statistics.pkl")
    with open(stats_file, 'wb') as f:
        pickle.dump({
            'mean': mean_np,
            'std': std_np,
            'feature_dim': feature_dim,
            'num_samples': total_sequences
        }, f)
    
    print(f"Statistics saved to:")
    print(f"  - {os.path.join(output_dir, 'rep15d_mean.npy')}")
    print(f"  - {os.path.join(output_dir, 'rep15d_std.npy')}")
    print(f"  - {stats_file} (with metadata)")
    
    return mean, std


# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate motion statistics")
    parser.add_argument("--root_path", type=str, required=True, help="Root path to the project")
    parser.add_argument("--cache_path", type=str, default="cache/", help="Path to cached data")
    parser.add_argument("--pose_rep", type=str, default="smplxflame_30", help="Pose representation type")
    
    args = parser.parse_args()
    
    mean, std = calculate_motion_statistics(args)
    
    # Print some summary statistics
    print("\nSummary Statistics:")
    print(f"Mean min/max/avg: {mean.min().item():.4f}/{mean.max().item():.4f}/{mean.mean().item():.4f}")
    print(f"Std min/max/avg: {std.min().item():.4f}/{std.max().item():.4f}/{std.mean().item():.4f}")