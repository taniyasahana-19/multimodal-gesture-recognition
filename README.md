
<div style="text-align: justify;">
Hand gesture recognition is a key component of vision-based human computer interaction with applications in sign language interpretation and assistive technologies. Despite advances in deep learning, the performance of convolutional neural networks (CNNs) varies considerably across architectures, datasets, and input modalities. In this study, we provide a systematic comparative evaluation of three widely used CNNs, VGG19, InceptionV3, and MobileNetV2, across six publicly available hand gesture datasets (ASL Fingerspelling, OUHANDS, JU-V2 Digit, JU-V2 Alphabet, NUS-II, and MUGD) using RGB, depth, and RGB-D inputs where applicable. Comprehensive data augmentation, including rotation, translation, shear, zoom, and horizontal flipping, is applied to enhance robustness. Model performance is quantitatively assessed using accuracy, precision, recall, and F-score, offering objective insights into the relative suitability of different CNN architectures and modalities for hand gesture recognition tasks.
</div>

## Datasets
- ASL Fingerspelling
- OUHANDS
- JU_V2_DIGIT
- JU_V2_ALPHA
- NUS‑II
- MUGD

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
- Accuracy, Precision, Recall, F‑Score

## Training Policy
- OUHANDS: Predefined train/test split
- Others: Runtime train/validation split

## Repository Structure
```text
gesture-recognition/
│
├── datasets/                  # Dataset directories (not included in repo)
│   ├── ASL/                   # ASL Fingerspelling (24 classes)
│   ├── OUHANDS/               # OUHANDS dataset (10 classes)
│   ├── JU_V2_DIGIT/           # JU-V2 Digit dataset (0–9)
│   ├── JU_V2_ALPHA/           # JU-V2 Alphabet dataset (24 classes)
│   ├── NUSII/                 # NUS-II dataset (10 classes)
│   └── MUGD/                  # MUGD dataset (36 alphanumeric classes)
│
├── models/                    # Deep learning architectures
│   ├── vgg19.py               # VGG19 model definition
│   ├── inceptionv3.py         # InceptionV3 model definition
│   └── mobilenetv2.py         # MobileNetV2 model definition
│
├── training/                  # Training and evaluation scripts
│   ├── trainer.py             # Training loop
│   ├── evaluator.py           # Accuracy, Precision, Recall, F1-score
│   └── plots.py               # Training vs Validation plots
│
├── utils/                     # Utility modules
│   ├── dataloader.py          # Dataset loading and runtime split logic
│   ├── augmentations.py       # Data augmentation techniques
│   └── config.py              # Dataset configuration and parameters
│
├── main.py                    # Main experiment runner
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
```

---

## Contributors

**Taniya Sahana**  
Research Scholar, Department of Computer Science and Engineering,  
Aliah University, Kolkata, India  

**Dr. Ayatullah Faruk Mollah**  
Assistant Professor, Department of Computer Science and Engineering,  
Aliah University, Kolkata, India 

---
##  Contact
- **Taniya Sahana**, Email: [taniyaswork@gmail.com](mailto:taniyaswork@gmail.com)
- **Dr. Ayatullah Faruk Mollah**, Email: [afmollah@aliah.ac.in](mailto:afmollah@aliah.ac.in)  
---
