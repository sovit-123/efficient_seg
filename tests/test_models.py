"""
USAGE:
python -m tests.test_models
"""

from models.effseg0_16s import EffSegModel

import torch

aux = False
model = EffSegModel(num_classes=2, aux=aux)
print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
model.eval()
tensor = torch.rand((1, 3, 512, 512))
output = model(tensor)
print(f"Output shape: {output['out'].shape}, {output['aux'].shape if aux else None}")