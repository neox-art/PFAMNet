# PFAMNet

This repository contains the PyTorch implementation of PFAMNet for cropland parcel delineation.

The model uses:

- PVTv2 backbone
- PFE: Parallel Feature Enhancement
- AMFF: Atrous-Enhanced Multi-level Feature Fusion
- Semantic segmentation, edge prediction, and distance regression heads
- Optional deep supervision for ablation experiments

The model entry point is:

```python
from models import PFAMNet

model = PFAMNet(backbone_weight_path='/workspace/pvt_v2_b2.pth')
```

Training and evaluation use `model_type='field'`. Configure the dataset,
backbone-weight, checkpoint, and output paths in the scripts for your environment.

Deep supervision can be controlled with:

```bash
python train1e4.py --use_deep_supervision true
python train1e4.py --use_deep_supervision false
```

Run the model tests with:

```bash
python -m unittest discover -s tests -v
```
