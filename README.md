# 🐦 Bird Species Classification - 524 Classes

State-of-the-art bird species classifier achieving **98.24% test accuracy** on 524 bird species using EfficientNetB0 with optimized training pipeline.

## 🏆 Performance

- **Test Accuracy:** 98.24%
- **Validation Accuracy:** 97.0%
- **Training Time:** 28 minutes (RTX 5070 Ti)
- **Model Size:** 30 MB
- **Classes:** 524 bird species

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Bird-Recognition-SOTA.git
cd Bird-Recognition-SOTA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install tensorflow pillow numpy
```

### Inference
```bash
# Predict on a single image
python predict.py path/to/bird_image.jpg
```

**Example output:**
```
📸 Image: test_image.jpg
Top 5 predictions:
--------------------------------------------------
1. AFRICAN FIREFINCH                        99.14%
2. SCARLET FACED LIOCICHLA                   0.48%
3. PYRRHULOXIA                               0.15%
```

## 📊 Dataset

- **Source:** [BIRDS 525 SPECIES](https://www.kaggle.com/datasets/vinjamuripavan/bird-species)
- **Total Images:** 89,885
  - Train: 84,635 images
  - Valid: 2,625 images
  - Test: 2,625 images
- **Classes:** 524 (LOONEY BIRDS excluded)
- **Image Size:** 224x224

## 🏗️ Architecture

- **Base Model:** EfficientNetB0 (ImageNet pretrained)
- **Head:** GlobalAveragePooling → Dropout(0.3) → Dense(524)
- **Training Strategy:**
  - Phase 1: Frozen base (15 epochs)
  - Phase 2: Fine-tuning last 30 layers (15 epochs)

## ⚡ Optimizations

- **Mixed Precision FP16:** 5-6x speedup
- **XLA JIT Compilation:** Enabled
- **TensorFlow Optimizations:**
  - oneDNN operations
  - TensorFloat-32 execution
  - cuDNN autotuning
- **Batch Size:** 512 (optimized for RTX 5070 Ti)

## 📈 Training Results

### Phase 1 (Frozen Base)
```
Epoch 15/15: train_acc=97.6%, val_acc=97.1%, loss=0.131
```

### Phase 2 (Fine-tuning)
```
Epoch 15/15: train_acc=97.5%, val_acc=97.0%, loss=0.117
```

### Test Evaluation
```
Test Accuracy: 98.24%
```

## 🛠️ Training from Scratch
```bash
# Download dataset (requires Kaggle API)
kaggle datasets download -d vinjamuripavan/bird-species -p data/raw --unzip

# Remove LOONEY BIRDS class
rm -rf data/raw/"BIRDS 525 SPECIES"/train/"LOONEY BIRDS"
rm -rf data/raw/"BIRDS 525 SPECIES"/test/"LOONEY BIRDS"
rm -rf data/raw/"BIRDS 525 SPECIES"/valid/"LOONEY BIRDS"

# Train model
python train.py
```

## 📁 Project Structure
```
Bird-Recognition-SOTA/
├── train.py                 # Training script
├── predict.py               # Inference script
├── config.py                # Configuration
├── data/
│   ├── raw/                 # Raw dataset
│   └── models/              # Saved models
│       └── bird_524_98.24percent.keras
├── checkpoints/             # Training checkpoints
└── logs/                    # Training logs
```

## 🔧 Requirements

- Python 3.12+
- TensorFlow 2.18+
- CUDA 12.2+ (for GPU training)
- 12+ GB VRAM (recommended for batch size 512)
- 32 GB RAM (recommended)

## 📝 License

MIT License

## 🙏 Acknowledgments

- Dataset: [vinjamuripavan/bird-species](https://www.kaggle.com/datasets/vinjamuripavan/bird-species)
- Base Model: EfficientNet (Google Research)
