import os
import pickle
import lmdb
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from Intentional_Gesture.dataloaders.beat_sep_lower import CustomDataset
from Intentional_Gesture.models.layers.modality_encoder import MultiModalEncoder

def main(args):
    # 1. Load dataset (which loads LMDB mapping)
    dataset = CustomDataset(args, loader_type=args.loader_type, build_cache=False)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # 2. Load original mapping file
    orig_mapping_path = os.path.join(dataset.preloaded_dir, "sample_db_mapping.pkl")
    with open(orig_mapping_path, "rb") as f:
        orig_mapping = pickle.load(f)
    orig_db_paths = orig_mapping['db_paths']
    orig_sample_mapping = orig_mapping['mapping']

    # 3. Prepare output LMDBs (mirror structure)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    new_db_paths = []
    envs_out = []
    for i, _ in enumerate(orig_db_paths):
        db_name = f"features_{i:02d}.lmdb"
        db_path = os.path.join(out_dir, db_name)
        new_db_paths.append(db_path)
        env = lmdb.open(db_path, map_size=20 * 1024 ** 3)
        envs_out.append(env)

    # 4. Load MultiModalEncoder (freeze params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MultiModalEncoder(args).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 5. Batch processing
    batch_size = args.batch_size
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # We'll keep a cache for each LMDB
    write_freq = 10
    cache = [{} for _ in envs_out]

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_samples)
        batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

        # Prepare batch tensors for MultiModalEncoder
        in_audio = torch.stack([s['audio_tensor'] for s in batch_samples]).to(device)
        cached_audio_low = torch.stack([s['audio_low'] for s in batch_samples]).to(device)
        cached_audio_high = torch.stack([s['audio_high'] for s in batch_samples]).to(device)
        word = torch.stack([s['word'] for s in batch_samples]).to(device)
        # If you use intention/semantic features, add them here as well

        with torch.no_grad():
            features = encoder(
                in_audio=in_audio,
                cached_audio_low=cached_audio_low,
                cached_audio_high=cached_audio_high,
                word=word
                # Add other args if needed
            )

        for i, idx in enumerate(range(batch_start, batch_end)):
            db_idx = orig_sample_mapping[idx]
            features_cpu = {k: (v[i].cpu().numpy() if v is not None else None) for k, v in features.items()}
            key = "{:08d}".format(idx).encode("ascii")
            cache[db_idx][key] = pickle.dumps(features_cpu)

        # Write to LMDB every write_freq batches or at the end
        if (batch_idx + 1) % write_freq == 0 or (batch_idx + 1) == num_batches:
            for db_idx, env in enumerate(envs_out):
                if cache[db_idx]:
                    with env.begin(write=True) as txn:
                        for k, v in cache[db_idx].items():
                            txn.put(k, v)
                    cache[db_idx] = {}

    # 6. Write new mapping file
    new_mapping = {
        "db_paths": new_db_paths,
        "mapping": orig_sample_mapping,
    }
    with open(os.path.join(out_dir, "sample_db_mapping.pkl"), "wb") as f:
        pickle.dump(new_mapping, f)

    print(f"Done. Saved new LMDBs and mapping to {out_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_path_1", type=str, required=True)
    parser.add_argument("--loader_type", type=str, default="train")
    parser.add_argument("--t_fix_pre", type=bool, default=False)
    parser.add_argument("--audio_dim", type=int, default=512)
    parser.add_argument("--audio_in", type=int, default=2)
    parser.add_argument("--raw_audio", type=bool, default=False)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--audio_fps", type=int, default=30)
    parser.add_argument("--use_exp", type=bool, default=False)
    parser.add_argument("--target_length", type=int, default=256)
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for new LMDBs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for precomputation")
    args = parser.parse_args()
    main(args)