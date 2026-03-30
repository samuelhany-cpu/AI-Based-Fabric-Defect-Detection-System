# Fabric Defect Detection - Phase 1

Phase 1 now uses the experiment 3 approach from the portfolio notebook: patch-level anomaly detection with a frozen `ResNet18` feature extractor and nearest-neighbor scoring. It is designed to be safe against data leakage, easy to run on a mid-range Windows laptop, and a better match for local texture defects than whole-image classification.

## What Phase 1 Includes
- dataset scanning and manifest generation
- reproducible train / validation / test split
- normal-only memory bank fitting for anomaly detection
- validation-based anomaly threshold calibration
- evaluation beyond accuracy
- single-image inference
- patch heatmap localization

## Expected Dataset Layout
The current scanner is configured for the dataset layout in `data/archive_3/`:

```text
data/archive_3/
|-- Defect_images/
|   |-- 0001_002_00.png
|   `-- ...
|-- Mask_images/
|   |-- 0001_002_00_mask.png
|   `-- ...
`-- NODefect_images/
    |-- 2306881-210020u/
    |   |-- 0001_000_05.png
    |   `-- ...
    `-- ...
```

Notes:
- Defect samples are grouped by the middle two filename tokens, for example `019_02`.
- Non-defect samples are grouped by their parent folder name.
- `group_id` is important because the split script keeps related images together.
- Defect masks are discovered automatically and added to the manifest when present.

## Setup
1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install a CUDA-enabled PyTorch build if you want GPU training locally.
4. Place the dataset under `data/archive_3/`.

## Local GPU Runtime
- The project runtime now defaults its Torch cache and temp files to `F:/Machine_Learning/.runtime-cache/`.
- Training will use CUDA automatically when `torch.cuda.is_available()` is `True`.
- If your `.venv` still reports a CPU-only build, reinstall PyTorch with CUDA support inside that environment before training.

## Commands
Create the split manifest:

```bash
python scripts/create_splits.py --config src/config/config.yaml
```

Fit the patch anomaly model:

```bash
python scripts/train.py --config src/config/config.yaml
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --config src/config/config.yaml --checkpoint outputs/models/best_model.pt --split test
```

Run inference on one image:

```bash
python scripts/predict.py --config src/config/config.yaml --checkpoint outputs/models/best_model.pt --image path/to/image.jpg
```

Generate a patch anomaly heatmap:

```bash
python scripts/explain.py --config src/config/config.yaml --checkpoint outputs/models/best_model.pt --image path/to/image.jpg
```

Run tests:

```bash
pytest
```

## Outputs
- `data/splits/manifest.csv`: manifest with labels, groups, and assigned split
- `outputs/models/`: checkpoints
- `outputs/metrics/`: metrics JSON files
- `outputs/figures/`: confusion matrices and anomaly heatmap images
- `outputs/logs/`: training and evaluation logs

## Current Assumptions
- Binary classes are still `defective` and `non_defective`, but training uses only `non_defective` images to build the memory bank.
- Related images should stay in the same split using archive-specific group IDs.
- The Phase 1 model uses `ResNet18` layer-3 patch embeddings with nearest-neighbor anomaly scoring.
- The anomaly threshold is calibrated from normal validation scores using a high quantile, not on test labels.
