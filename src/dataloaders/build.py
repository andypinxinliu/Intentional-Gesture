from dataloaders.beat_sep_lower import build_beat_sep_dataset
from dataloaders.mix_sep import build_beat_rvq_dataset

def build_dataset(args, **kwargs):
    if args.dataset == 'beat_rvq':
        return build_beat_rvq_dataset(args, **kwargs)
    if args.dataset == 'beat_sep':
        return build_beat_sep_dataset(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')