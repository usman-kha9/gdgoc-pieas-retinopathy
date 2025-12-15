```markdown
# GDGOC-PIEAS-AI-ML-Hackathon-2025 — Diabetes Detection through Retinopathy

Overview
This project is a ready-to-run PyTorch pipeline to train a custom CNN (from scratch) to classify retinal images into five classes:
- 0: No DR
- 1: Mild DR
- 2: Moderate DR
- 3: Severe DR
- 4: Proliferative DR

Key improvements in this final version
- Class rebalancing (class weights + WeightedRandomSampler)
- Optional Focal Loss
- Mixed-precision training (AMP)
- TensorBoard logging and confusion matrix visualization
- Single-image inference script with Grad-CAM overlay helper
- Better checkpointing and model export

Dataset
Download dataset:
https://www.kaggle.com/datasets/kushagratandon12/diabetic-retinopathy-balanced/data

Project structure
├── team_name/
│   ├── src/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── utils.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── notebooks/
│   │   └── README.md
│   ├── REPORT_TEMPLATE.md
│   ├── requirements.txt
│   └── README.md

Quick start
1. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac

2. Install dependencies
   pip install -r requirements.txt

3. Prepare dataset
- Produce a metadata CSV with at least columns `image_path` and `label` (labels 0-4). Paths may be relative to `--img_dir`.

4. Train (example)
   python src/train.py \
     --csv_path path/to/metadata.csv \
     --img_dir path/to/images \
     --output_dir outputs \
     --epochs 40 \
     --batch_size 32 \
     --lr 1e-3 \
     --use_weighted_sampler True \
     --use_focal False

Options of interest
- --use_weighted_sampler True/False : enable WeightedRandomSampler to balance batches
- --use_focal True/False : use FocalLoss instead of CrossEntropyLoss
- --amp True/False : enable automatic mixed precision (default True if CUDA available)
- --image_size : resize images
- --tensorboard_dir : where to write TensorBoard logs

Inference (single image)
   python src/inference.py \
     --model_path outputs/final_best.pth \
     --img_path /path/to/sample.jpg \
     --img_dir /path/to/images \
     --image_size 224 \
     --use_gradcam True

Notes
- This project trains from scratch (no pretrained weights) to comply with hackathon rules.
- For class imbalance, try both WeightedRandomSampler and class-weighted loss; FocalLoss can help with hard-to-classify minority classes.

If you'd like, I can push this project to GitHub for you — tell me the repository (owner/repo) or say "create new repo under my account named <repo-name>" and confirm branch name and whether to open a PR.
```