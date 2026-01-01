
# Multi‑Dataset Hand Gesture Recognition (PyTorch)

## Datasets
- ASL Fingerspelling (24 classes)
- OUHANDS (10 classes)
- JU_V2_DIGIT (0–9)
- JU_V2_ALPHA (24 classes)
- NUS‑II (10 classes)
- MUGD (36 classes)

## Models
- VGG19
- InceptionV3
- MobileNetV2

## Modalities
- RGB (3‑channel)
- Depth (1‑channel)
- RGB‑D (4‑channel)

## Augmentation
- Rotation (±40°)
- Width / Height Shift (20%)
- Shear
- Zoom (20%)
- Horizontal Flip

## Metrics
Accuracy, Precision, Recall, F1‑Score (4 decimals)

## Training Policy
- OUHANDS: Predefined train/test split
- Others: Runtime train/validation split

Run:
```bash
python main.py --dataset ASL --model mobilenetv2 --mode rgb
```
