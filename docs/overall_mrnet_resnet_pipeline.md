# Overall MRNet-Style ResNet Pipeline

## Objective

Build a knee MRI abnormality-detection pipeline that:

1. reads DICOM MRI slices,
2. preprocesses each slice,
3. uses a pretrained ResNet to extract visual features,
4. combines information across all slices,
5. predicts study- or series-level abnormalities.

---

## High-Level Pipeline

```text
MRI STUDY / SERIES
        │
        ├── DICOM 1
        ├── DICOM 2
        ├── DICOM 3
        ├── ...
        └── DICOM N
              │
              ▼
      PREPROCESSING
              │
      ├── read DICOM
      ├── convert to float
      ├── normalize intensity
      ├── resize
      ├── convert to 3 channels
      └── tensor normalization
              │
              ▼
      DATA AUGMENTATION
       training only
              │
      ├── small rotation
      ├── small translation
      ├── small scaling
      └── mild intensity variation
              │
              ▼
       PRETRAINED RESNET-18
      applied to every slice
              │
              ▼
      512 FEATURES PER SLICE
              │
        N slices × 512
              │
              ▼
       SLICE AGGREGATION
              │
      max across slices
              │
              ▼
       512 FEATURES FOR
        ENTIRE MRI SERIES
              │
              ▼
          CLASSIFIER
        Linear(512 → 12)
              │
              ▼
          12 LOGITS
              │
              ▼
           SIGMOID
              │
              ▼
    12 ABNORMALITY PROBABILITIES
```

---

## Stage 1: DICOM Loading

Each DICOM represents one slice from an MRI series.

The pipeline loads the pixel array and converts it into a floating-point image.

```text
DICOM
  ↓
2D MRI image
```

---

## Stage 2: Preprocessing

Each slice is made numerically consistent.

Typical baseline:

```text
MRI slice
  ↓
intensity normalization
  ↓
resize to 224 × 224
  ↓
repeat grayscale into 3 channels
  ↓
tensor normalization
```

Output:

```text
3 × 224 × 224
```

---

## Stage 3: Augmentation

During training only, apply conservative random transformations.

Examples:

```text
small rotation
small translation
small zoom
mild intensity changes
```

The objective is to reduce overfitting while preserving realistic knee anatomy.

Validation and test images should not receive random augmentation.

---

## Stage 4: ResNet-18 Feature Extraction

The pretrained ResNet processes each slice independently.

The ImageNet classifier is removed:

```python
model.fc = nn.Identity()
```

Each slice becomes:

```text
512 learned numerical features
```

For `N` slices:

```text
N × 512
```

---

## Stage 5: Slice Aggregation

The model needs one representation for the complete series.

Baseline:

```text
N × 512
    ↓
max pooling across slices
    ↓
1 × 512
```

This retains the strongest response for every feature dimension.

It is especially useful when an abnormality is strongly visible in only a few slices.

---

## Stage 6: Classification

The aggregated 512-dimensional representation enters a classifier:

```python
nn.Linear(512, 12)
```

The classifier outputs:

```text
12 logits
```

During training:

```text
logits
  ↓
BCEWithLogitsLoss
```

During inference:

```text
logits
  ↓
sigmoid
  ↓
12 probabilities
```

---

## Stage 7: Training

A training batch conceptually contains:

```text
MRI series
    +
12 binary target labels
```

Forward pass:

```text
series
  ↓
preprocessing
  ↓
augmentation
  ↓
ResNet features
  ↓
slice aggregation
  ↓
classifier
  ↓
logits
```

Loss:

```text
logits + targets
      ↓
BCEWithLogitsLoss
      ↓
loss
```

Backpropagation updates the trainable model parameters.

---

## Stage 8: Validation

Validation uses the same core pipeline without random augmentation.

Track:

- validation loss,
- ROC AUC for each abnormality,
- competition metric,
- class-specific failures.

Use study-level splits so slices from the same study cannot leak between training and validation.

---

## Initial Recommended Architecture

```text
Backbone:
    ImageNet-pretrained ResNet-18

Input:
    224 × 224, 3-channel repeated grayscale

Feature size:
    512 per slice

Aggregation:
    max pooling across slices

Classifier:
    Linear(512 → 12)

Loss:
    BCEWithLogitsLoss

Output:
    12 sigmoid probabilities
```

---

## Development Path

### Baseline

```text
ResNet-18
+
max pooling
+
linear classifier
```

Get the full pipeline working first.

### Then Test Improvements

```text
Preprocessing
    ↓
Augmentation
    ↓
Slice sampling
    ↓
Aggregation method
    ↓
ResNet-50
    ↓
Class weighting
    ↓
Ensembling
```

Only change one major component at a time when possible.

---

## Future Architecture Experiments

Once the baseline is stable, compare:

### Backbone

```text
ResNet-18
vs
ResNet-34
vs
ResNet-50
```

### Aggregation

```text
max pooling
vs
mean pooling
vs
max + mean pooling
vs
attention pooling
```

### Classifier

```text
linear
vs
small MLP
```

### Ensemble

Possible later ensemble:

```text
ResNet-18 predictions
          +
ResNet-50 predictions
          ↓
weighted average
          ↓
final probabilities
```

The ensemble should only be retained if it improves out-of-fold or validation performance.

---

## Mental Model

The three core components have different jobs.

### ResNet

```text
"What visual patterns are present in this slice?"
```

### Aggregation

```text
"What information is present across the full MRI stack?"
```

### Classifier

```text
"Given that information, how likely is each abnormality?"
```

Together:

```text
DICOM slices
    ↓
ResNet
    ↓
slice features
    ↓
aggregation
    ↓
series representation
    ↓
classifier
    ↓
abnormality probabilities
```
