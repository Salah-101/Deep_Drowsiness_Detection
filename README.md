# 😴 Driver Drowsiness Detection

A binary image classification project that detects whether a driver is **Awake** or **Drowsy** using two different deep learning approaches: a custom CNN built from scratch and a Transfer Learning model based on MobileNetV2.

---

## 📁 Project Structure

```
├── project_from_scratch_2.ipynb        # CNN trained from scratch
└── Project_image_Transfer_learning.ipynb  # Transfer Learning with MobileNetV2
```

---

## 🗂️ Dataset

The dataset is organized into three splits stored on Google Drive:

```
MyDrive/
├── train/
│   ├── awake/
│   └── drowsy/
├── valid/
│   ├── awake/
│   └── drowsy/
└── test/
    ├── awake/
    └── drowsy/
```

| Class | Label |
|-------|-------|
| Awake | 0 |
| Drowsy | 1 |

---

## 🔬 Approach 1 — CNN from Scratch

> **Notebook:** `project_from_scratch_2.ipynb`

### Preprocessing Pipeline

1. Read image using OpenCV
2. Convert to **grayscale**
3. Resize to **64 × 64**
4. Normalize pixel values to **[0, 1]**
5. Save processed dataset as `.npz`

### Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D + MaxPool | 32 filters, (3×3), ReLU |
| Conv2D + MaxPool | 64 filters, (3×3), ReLU |
| Conv2D + MaxPool | 64 filters, (3×3), ReLU |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Dense (Output) | 1 unit, Sigmoid |

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Input Shape:** (64, 64, 1)

### Data Augmentation

Applied during training using `ImageDataGenerator`:
- Rotation range: ±10°
- Zoom range: 10%
- Horizontal flip

### Training

- **Epochs:** 100
- **Batch Size:** 32

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC / AUC Curve
- Full Classification Report

---

## 🚀 Approach 2 — Transfer Learning (MobileNetV2)

> **Notebook:** `Project_image_Transfer_learning.ipynb`

### Model Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet, `include_top=False`)
- **Input Shape:** (96, 96, 3)
- **Custom Head:** GlobalAveragePooling2D → Dense(64, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- **API Style:** Functional API

### Two-Phase Training Strategy

**Phase 1 — Head Training**
- Freeze all base model layers
- Train only the custom classification head
- Epochs: 10 | Learning Rate: 1e-3
- Callbacks: EarlyStopping (patience=3), ReduceLROnPlateau

**Phase 2 — Fine-Tuning**
- Unfreeze the last **20 layers** of MobileNetV2
- Re-train with a much smaller learning rate
- Epochs: 10 | Learning Rate: 1e-5
- Callbacks: EarlyStopping (patience=4), ModelCheckpoint (saves best model)

### Evaluation Metrics

- Test Accuracy & Loss
- Confusion Matrix
- ROC / AUC Curve
- Accuracy & Loss curves (both phases combined)
- Full Classification Report

---

## 📊 Results Comparison

| Metric | CNN from Scratch | Transfer Learning (MobileNetV2) |
|--------|:-:|:-:|
| Accuracy | — | — |
| Precision | — | — |
| Recall | — | — |
| F1-Score | — | — |
| AUC | — | — |

> ⚠️ Fill in the actual results after running the notebooks.

---

## 🛠️ Requirements

```bash
tensorflow >= 2.x
numpy
opencv-python
matplotlib
seaborn
scikit-learn
google-colab  # for Drive mounting
```

Install via:

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn
```

---

## ▶️ How to Run

1. Upload the notebooks to **Google Colab**
2. Mount your Google Drive and ensure the dataset is placed at the correct paths (`MyDrive/train`, `MyDrive/valid`, `MyDrive/test`)
3. Run all cells in order

---

## 💾 Saved Outputs

| File | Description |
|------|-------------|
| `dataset.npz` | Preprocessed images + labels (Approach 1) |
| `drowsiness_model.keras` | Best saved model weights |
| `history.json` | Training history for both phases (Approach 2) |
| `preview.png` | Before/after preprocessing samples |
| `confusion_matrix.png` | Confusion matrix plot |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).