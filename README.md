# Fabric Defect Detection - Phase 1

Phase 1 is a learning-first image-classification pipeline for binary textile defect detection. It is designed to be safe against data leakage, easy to run on a mid-range Windows laptop, and ready to extend once the AITEX dataset is available locally.

## What Phase 1 Includes
- dataset scanning and manifest generation
- reproducible train / validation / test split
- binary classifier training with transfer learning
- evaluation beyond accuracy
- single-image inference
- Grad-CAM explainability

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

3. Place the dataset under `data/archive_3/`.

## Commands
Create the split manifest:

```bash
python scripts/create_splits.py --config src/config/config.yaml
```

Train the baseline:

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

Generate a Grad-CAM heatmap:

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
- `outputs/figures/`: confusion matrices, learning curves, Grad-CAM images
- `outputs/logs/`: training and evaluation logs

## Current Assumptions
- Binary classes are `defective` and `non_defective`.
- Related images should stay in the same split using archive-specific group IDs.
- The first baseline uses `ResNet18` with ImageNet pretrained weights.
- Deployment is intentionally deferred until after the classifier is working well.
