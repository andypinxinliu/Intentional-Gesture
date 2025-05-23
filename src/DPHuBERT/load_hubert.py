import torch
from wav2vec2.model import wav2vec2_model

ckpt_path = "./DPWavLM-sp0.75.pth"
ckpt = torch.load(ckpt_path)
model = wav2vec2_model(**ckpt["config"])
result = model.load_state_dict(ckpt["state_dict"], strict=False)
print(f"missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")
print(f"{sum(p.numel() for p in model.parameters())} params")

breakpoint()