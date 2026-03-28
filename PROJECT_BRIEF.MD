# Project Brief - AI-Based Fabric Defect Detection System

## 1) Project Title
AI-Based Fabric Defect Detection System for Automated Textile Quality Inspection

---

## 2) Project Summary
Build a portfolio-grade computer vision system that detects defects in textile fabric images and classifies fabric as **defective** or **non-defective**, with optional visual explanation of suspicious regions.

This project should be treated as a serious ML engineering project, but the implementation should stay learning-focused. The first goal is to build a reliable binary image-classification pipeline end-to-end. Deployment features and advanced modeling should come only after the core classification workflow is working well.

The project should demonstrate:
- proper dataset handling
- reproducible training
- evaluation beyond plain accuracy
- explainability
- clean engineering structure suitable for GitHub portfolio use

Deployment is valuable, but it is a Phase 2 goal rather than part of the first MVP.

---

## 3) Problem Statement
In textile manufacturing, visual inspection is critical for maintaining product quality. Manual inspection is labor-intensive and error-prone, especially under high throughput. Fabric defects such as holes, stains, broken yarns, misweaves, and surface irregularities can lead to production waste, customer complaints, and financial loss.

The objective is to automate first-pass visual inspection using computer vision so that the system can:
1. identify whether a fabric sample is defective
2. optionally categorize the defect type
3. optionally highlight suspicious image regions to support operator review

---

## 4) Product Goal
Develop a practical ML system that can assist textile QA teams by screening fabric images and flagging likely defects with confidence scores and visual explanations.

For the first version, success means building a trustworthy binary defect classifier with reproducible training, proper evaluation, and Grad-CAM-based explanation.

---

## 5) Recommended Project Scope
### Phase 1 (must-have, learning-first MVP)
Build a **binary classifier**:
- input: fabric image
- output: defective / non-defective
- provide confidence score
- provide Grad-CAM heatmap for explainability
- support reproducible train / validation / test split
- report strong evaluation metrics and failure cases

### Phase 2 (portfolio upgrade)
Add productization features:
- FastAPI inference service
- Streamlit demo app
- upload image -> show prediction + confidence + explanation map

### Phase 3 (strong ML upgrade)
Add **multi-class defect classification** if labels support it:
- hole
- stain
- broken yarn
- misweave
- other surface defects

### Phase 4 (advanced / standout)
Add **defect localization or anomaly detection**:
- weak localization via Grad-CAM
- anomaly heatmap or segmentation-style output if feasible
- lightweight anomaly-detection baseline for comparison

---

## 6) Why This Project Is Strong for a Portfolio
This project demonstrates:
- computer vision fundamentals
- transfer learning
- industrial inspection / defect detection use case
- practical ML engineering
- explainability
- realistic evaluation discipline

It is especially strong because it targets a real manufacturing problem rather than a beginner dataset like cats vs dogs.

---

## 7) Target Users
- textile factory quality control teams
- production supervisors
- small garment manufacturers
- internal operators reviewing fabric rolls or fabric samples

---

## 8) Recommended Dataset Strategy
### Primary dataset
Use the **AITEX Fabric Image Database** as the main textile-specific dataset. It is a strong fit for the project story, but it is relatively small, so the pipeline should rely on transfer learning, careful splitting, and moderate augmentation.

### Secondary dataset / backup / benchmarking
Use **MVTec AD** only as a secondary benchmark or reference for anomaly-detection framing. It should not distract from the main supervised binary-classification pipeline.

### Optional fallback for experimentation
The **NEU surface defect database** is not textile-specific, but it can be useful for extra classification experiments if needed.

### Default recommendation
Use:
1. AITEX for the final portfolio story
2. MVTec AD concepts for future anomaly-detection framing
3. NEU only if additional experiments are needed

### Required data decisions before coding
Before implementation, inspect the real dataset and answer these questions:
1. What is one sample: a full image, a crop, or a patch?
2. How will labels map into `defective` and `non_defective`?
3. What metadata identifies related samples from the same source fabric or acquisition session?
4. How will train / validation / test splits avoid leakage?

### Recommended label strategy
For the MVP:
- use binary classification only
- map all defect categories into a single `defective` class
- keep any finer defect labels only for future multi-class experiments
- if masks or localization annotations exist, save them for analysis and later extensions rather than making them mandatory for the first training target

### Recommended split strategy
Do **not** rely on a naive random split if related samples come from the same fabric roll, image strip, or source group.

Preferred strategy:
- split by fabric source or other grouping metadata when available
- if grouped metadata is not explicitly provided, infer the safest grouping possible from filenames or folder structure
- make the split reproducible with a fixed seed
- document all assumptions clearly

This is one of the most important design decisions in the entire project because careless splitting can create data leakage and misleadingly strong metrics.

---

## 9) Machine Learning Framing
Evaluate two approaches, but implement one fully before adding the other.

### Approach A - Supervised transfer learning
Best if labels are available.
- backbone: ResNet18 / EfficientNet-B0 / MobileNetV3
- fine-tune on binary defect classification
- use weighted loss if classes are imbalanced
- use image augmentations carefully

### Approach B - Anomaly detection
Best if defect-free images dominate or labels are weak.
- train primarily on normal images
- detect anomalous patterns at inference
- possible methods:
  - autoencoder baseline
  - feature reconstruction baseline
  - patch-based anomaly approach
  - simple embedding distance baseline

### Recommended final choice
Start with **Approach A: supervised transfer learning** because it is easier to complete cleanly on current hardware, easier to evaluate, and better aligned with learning image classification fundamentals.

Add a lightweight anomaly-detection experiment only after the supervised baseline is stable.

---

## 10) Hardware Constraints
Local machine:
- RTX 3050 Ti with 4GB VRAM
- 8GB RAM
- Windows laptop environment

Therefore the implementation must be optimized for limited hardware:
- use transfer learning
- image size around 224x224 by default
- batch size around 8 to 16 depending on memory
- mixed precision if supported
- avoid very large backbones
- keep training modular so it can later move to Kaggle or Colab if needed

---

## 11) Functional Requirements
The Phase 1 MVP must:
1. load and preprocess fabric image datasets
2. support train / validation / test split
3. train a binary classifier for defect detection
4. evaluate with:
   - accuracy
   - precision
   - recall
   - F1
   - ROC-AUC if applicable
   - confusion matrix
5. save best model checkpoints
6. run inference on a single image
7. output:
   - predicted label
   - confidence score
   - explanation heatmap
8. log experiments and metrics
9. support configuration-driven training

Phase 2 may add:
1. API-based inference
2. demo UI
3. batch inference

---

## 12) Non-Functional Requirements
The codebase must be:
- modular
- reproducible
- readable
- portfolio-ready
- documented
- easy to run locally

Engineering expectations:
- use `.env.example` if needed
- use `requirements.txt` or `pyproject.toml`
- include CLI commands or a simple task runner
- save outputs under structured folders
- add basic tests for non-training utilities

Avoid overengineering. Simpler, reliable code is better than a large architecture that is hard to debug.

---

## 13) Deliverables
### Phase 1 deliverables
#### A. ML pipeline
- data loader
- training script
- evaluation script
- inference script
- explainability script

#### B. Documentation
- polished README
- architecture overview
- dataset notes
- training instructions
- evaluation summary
- limitations and future work

#### C. Assets
- sample prediction screenshots
- confusion matrix image
- Grad-CAM examples
- metrics report

### Phase 2 deliverables
#### A. Deployment layer
- FastAPI app with `/predict`
- optional `/health`
- optional `/predict-batch`

#### B. Demo
- Streamlit app with image upload and visualization

---

## 14) Suggested Repository Structure
```text
fabric-defect-detection/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- splits/
|
|-- notebooks/
|   |-- 01_eda.ipynb
|   `-- 02_model_experiments.ipynb
|
|-- scripts/
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- explain.py
|
|-- src/
|   |-- config/
|   |   `-- config.yaml
|   |-- data/
|   |   |-- dataset.py
|   |   |-- transforms.py
|   |   `-- split.py
|   |-- models/
|   |   |-- classifier.py
|   |   `-- factory.py
|   |-- training/
|   |   |-- engine.py
|   |   |-- loss.py
|   |   `-- metrics.py
|   |-- inference/
|   |   |-- predict.py
|   |   `-- explain.py
|   `-- utils/
|       |-- logger.py
|       `-- seed.py
|
|-- app/
|   `-- streamlit_app.py
|
|-- api/
|   `-- main.py
|
|-- outputs/
|   |-- models/
|   |-- metrics/
|   |-- figures/
|   `-- logs/
|
|-- tests/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

Note:
- `api/` and `app/` are Phase 2 directories
- do not let deployment code block the Phase 1 ML pipeline

---

## 15) Data Handling Expectations
Implementation must:
- inspect the dataset structure first
- create a reproducible split
- handle class imbalance
- apply augmentations only to training data
- avoid data leakage
- support configurable dataset paths
- document the final label mapping
- document the final split logic

Suggested augmentations:
- resize
- random horizontal flip if valid for texture orientation
- small rotation if valid
- color jitter only if it does not distort defect cues
- normalization using ImageNet stats for pretrained backbones

Important caution:
- if flips or rotations change the meaning of textile patterns or defects, disable them
- prioritize realistic augmentations over aggressive augmentation volume

---

## 16) Modeling Requirements
### Minimum baseline
- model: ResNet18 first
- pretrained weights
- final head adapted for binary classification

### Recommended progression
1. implement ResNet18 baseline
2. verify the entire training and evaluation loop
3. inspect failure cases
4. only then try EfficientNet-B0 or MobileNetV3 if useful

### Training setup
- optimizer: AdamW
- scheduler: ReduceLROnPlateau or cosine schedule
- early stopping
- checkpointing on best validation F1
- mixed precision if available

### Evaluation
Must include:
- train / validation curves
- confusion matrix
- per-class precision / recall / F1
- failure-case analysis

### Metric emphasis
Accuracy should not be the headline metric if classes are imbalanced.

Special attention should be given to:
- defective-class recall
- defective-class F1
- whether the model beats simple baselines such as majority-class prediction

---

## 17) Explainability Requirements
The system must generate a visual explanation for predictions:
- implement Grad-CAM or equivalent heatmap
- overlay the heatmap on the original image
- save examples for correct and incorrect predictions
- include a short note in the README explaining that heatmaps assist review but are not pixel-accurate localization

Important framing:
- Grad-CAM is part of the MVP as an explainability tool
- it should not be presented as precise defect localization

---

## 18) API Requirements
This is a **Phase 2** requirement, not a blocker for the MVP.

Build a FastAPI service with:
- `GET /health`
- `POST /predict`

`POST /predict` should:
- accept an image file
- preprocess image
- run model inference
- return JSON containing:
  - predicted_label
  - confidence
  - class_probabilities if available
  - heatmap path or encoded response if practical

Optional:
- `POST /predict-batch`

---

## 19) Demo App Requirements
This is a **Phase 2** requirement, not a blocker for the MVP.

Build a Streamlit app that:
- allows image upload
- shows original image
- shows prediction
- shows confidence
- shows Grad-CAM heatmap
- explains in plain language what the result means

---

## 20) Success Criteria
The Phase 1 project is considered complete when:
1. the dataset is inspected and documented
2. the label strategy is clearly defined
3. the split is reproducible and leakage-aware
4. the model trains end-to-end locally
5. inference works on unseen images
6. evaluation includes more than just accuracy
7. at least one explainability method is implemented
8. the README is polished enough for recruiters

Phase 2 success adds:
1. API returns valid predictions
2. demo app works locally

---

## 21) Acceptance Criteria
### Must-have for MVP
- reproducible training pipeline
- binary classification on textile images
- documented label strategy
- leakage-aware split strategy
- saved best checkpoint
- evaluation report
- inference on unseen images
- Grad-CAM
- clean README

### Nice-to-have after MVP
- FastAPI inference endpoint
- Streamlit demo
- multi-class defect labels
- anomaly-detection baseline
- Dockerfile
- unit tests
- ONNX export
- batch inference

---

## 22) Risks and Constraints
- dataset may be small
- labels may be inconsistent
- careless splitting may create leakage
- binary classification may be easier than precise localization
- anomaly detection is more realistic industrially but increases implementation complexity
- local GPU memory is limited

Mitigation:
- use transfer learning
- start with binary classification
- keep image size moderate
- prefer group-aware splits over naive random splits
- structure project so heavier experiments can move to cloud later

---

## 23) Final Positioning for Portfolio / README
Present the project as:

**AI-Based Fabric Defect Detection System**

A computer vision system for automated textile quality inspection using transfer learning and explainable AI. Built to classify defective vs non-defective fabric samples, visualize defect evidence through Grad-CAM, and support future deployment through an API and demo interface.

This wording is honest and strong because it highlights the real completed system without forcing deployment work into the first milestone.

---

## 24) Implementation Priority Order
1. inspect dataset structure
2. define final binary label strategy
3. define leakage-aware split strategy
4. scaffold the repository
5. implement data pipeline
6. implement baseline model training
7. run evaluation pipeline
8. implement inference
9. implement Grad-CAM
10. write README and usage instructions
11. identify next improvements after the MVP
12. add FastAPI and Streamlit only after the MVP is stable

---

## 25) Instructions to Codex
Continue this project as an implementation-focused ML engineering task. Work incrementally and produce production-style, readable code.

Execution instructions:
1. inspect dataset structure and propose final label strategy
2. define and document the split strategy before training
3. scaffold the repository
4. implement data pipeline
5. implement baseline model training
6. run evaluation pipeline
7. implement inference and Grad-CAM
8. write README and usage instructions
9. identify next improvements after the MVP
10. add API and demo only after the core ML pipeline is complete

Important constraints:
- optimize for limited GPU memory
- prefer simple robust solutions over overengineered ones
- keep all hyperparameters configurable
- ensure the project runs locally on Windows
- explain any assumptions before making major dataset-specific decisions
- do not allow deployment work to delay the first end-to-end classification baseline

---

## 26) Optional Stretch Goals
- multi-class defect classification
- semi-supervised defect detection
- anomaly score threshold tuning
- defect region segmentation
- webcam / real-time inspection mode
- ONNX or TensorRT export
- Dockerized deployment
- experiment tracking with MLflow
- CI workflow for lint + tests
