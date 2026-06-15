# CIFAR-10 CNN Classifier

A CNN built from scratch for 10-class image classification on CIFAR-10.
No pretrained weights — every component (convolution, training loop, evaluation) is implemented manually.

## Results

| Version | Change | Test Accuracy |
|---------|--------|---------------|
| v1 Baseline | 2-layer CNN + FC | 62% |
| v2 MaxPooling | Pooling after each conv | 65% |
| v3 Augmentation + 20 epochs | RandomFlip + RandomCrop | **69%** |

## Architecture

```
Input: 3×32×32

Conv1(3→16, k=3) → ReLU → MaxPool(2×2)    # 16×15×15
Conv2(16→32, k=3) → ReLU → MaxPool(2×2)   # 32×6×6
Flatten → FC(1152→10)
```

Output size derivation:
- Conv2d(kernel=3, no padding): `(32-2)/2 = 15`
- After second Conv + Pool: `(15-2)/2 = 6`
- Flatten: `32 × 6 × 6 = 1152`

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cifar10-cnn.git
cd cifar10-cnn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train (dataset downloads automatically)
python train.py
```

~20 min on CPU, ~3 min on GPU.

## Project Structure

```
cifar10-cnn/
├── src/
│   ├── model.py      # SimpleCNN (nn.Module)
│   └── dataset.py    # Data loading + augmentation
├── train.py          # Training & evaluation
├── results/          # Saved model weights (.pth)
├── requirements.txt
└── .gitignore
```

## Details

**Data augmentation** (training set only)
- `RandomHorizontalFlip` — prevents the model from relying on object orientation
- `RandomCrop(32, padding=4)` — pads to 40×40 then crops back to 32×32, simulating positional shift
- `Normalize(0.5, 0.5, 0.5)` — maps pixels from [0,1] to [-1,1] for faster convergence

**Training config**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch size: 64
- Epochs: 20

## What I Learned

- Writing `nn.Module` by hand and understanding how `forward()` executes
- Why `optimizer.zero_grad()` must be called every step
- How MaxPooling reduces spatial size and expands the receptive field
- The real impact of data augmentation on generalisation (+7% accuracy)
- Why `torch.no_grad()` saves memory during inference

## Next Steps

- [ ] Add BatchNorm and observe training stability
- [ ] Compare SGD vs Adam convergence curves
- [ ] Implement ViT (Vision Transformer) on the same task

---

*UTS Bachelor of AI — Year 1*
