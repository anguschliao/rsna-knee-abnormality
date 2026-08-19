# MRNet Study Summary

## Paper

**Bien, N., Rajpurkar, P., Ball, R. L., et al. (2018).
*Deep-learning-assisted diagnosis for knee magnetic resonance imaging:
Development and retrospective validation of MRNet.* PLOS Medicine,
15(11), e1002699.**

MRNet was developed by researchers at Stanford to test whether deep
learning could automatically detect clinically important findings from
knee MRI examinations and whether its predictions could help physicians
interpret MRI scans.

------------------------------------------------------------------------

## 1. Study Objective

The study had three main goals:

1.  Develop a deep-learning system capable of detecting pathology
    directly from knee MRI examinations.
2.  Compare the model's diagnostic performance with practicing
    radiologists and orthopedic surgeons.
3.  Determine whether providing the model's predictions to clinicians
    could improve their diagnostic performance.

The model focused on three binary classification tasks:

-   **General abnormality**
-   **Anterior cruciate ligament (ACL) tear**
-   **Meniscal tear**

Rather than trying to identify every possible knee diagnosis, the study
demonstrated that a neural network could process an entire MRI
examination and produce clinically useful probabilities for selected
abnormalities.

------------------------------------------------------------------------

## 2. Dataset

The Stanford dataset contained **1,370 knee MRI examinations** performed
between 2001 and 2012.

The dataset included:

  Finding           Exams   Prevalence
  --------------- ------- ------------
  Abnormal          1,104        80.6%
  ACL tear            319        23.3%
  Meniscal tear       508        37.1%

Some examinations contained more than one diagnosis. For example, ACL
and meniscal tears could occur in the same knee.

Each MRI examination contained multiple imaging series acquired in
different anatomical planes and with different MRI sequences.

The routine protocol included sequences such as:

-   coronal T1
-   coronal T2 with fat saturation
-   sagittal proton-density (PD)
-   sagittal T2 with fat saturation
-   axial PD with fat saturation

The labels were initially extracted from clinical radiology reports. For
the internal validation set, three musculoskeletal radiologists
independently reviewed the examinations, and their majority decision was
used as the reference standard.

------------------------------------------------------------------------

## 3. Why MRI Requires a Different Approach from Ordinary Image Classification

An MRI examination is not a single image.

A series contains a **variable number of 2D slices** through the knee.
Therefore, a model must solve two problems:

1.  Extract useful features from each individual slice.
2.  Combine information across all slices to classify the entire MRI
    series.

MRNet handled this by applying the same CNN to every slice and then
aggregating the resulting slice-level features.

This is an important design idea for our RSNA project because the number
of DICOM slices can also vary between series.

------------------------------------------------------------------------

## 4. Image Preprocessing

Each 2D MRI slice was prepared for the neural network and represented as
a three-channel image.

The model input for one MRI series had the form:

``` text
s × 3 × 256 × 256
```

where:

-   `s` = number of slices in the MRI series
-   `3` = image channels
-   `256 × 256` = spatial image dimensions

The key point is that **the model did not require every MRI series to
contain the same number of slices**.

------------------------------------------------------------------------

## 5. MRNet Architecture

The core MRNet pipeline was:

``` text
MRI series
    ↓
Individual 2D slices
    ↓
AlexNet feature extractor applied to every slice
    ↓
Feature maps for every slice
    ↓
Global average pooling within each slice
    ↓
One feature vector per slice
    ↓
Max pooling across slices
    ↓
One feature vector representing the entire series
    ↓
Fully connected layer
    ↓
Sigmoid
    ↓
Probability of diagnosis
```

### Slice Feature Extraction

MRNet used an **AlexNet-based CNN**, pretrained on ImageNet, as its
feature extractor.

Each slice was passed independently through the CNN.

For a series containing `s` slices, the CNN produced approximately:

``` text
s × 256 × 7 × 7
```

features.

This means that every slice generated 256 feature maps.

### Global Average Pooling

MRNet then averaged each 7 × 7 feature map:

``` text
s × 256 × 7 × 7
        ↓
s × 256
```

Each slice was therefore represented by a **256-dimensional feature
vector**.

### Max Pooling Across Slices

The model then needed to turn a variable number of slice vectors into
one fixed-length vector.

It performed element-wise maximum pooling across the slice dimension:

``` text
s × 256
   ↓
256
```

For each of the 256 learned features, MRNet retained the strongest
activation found anywhere in the MRI series.

Conceptually:

> "Did any slice in this MRI strongly contain this feature?"

This is one of the most important ideas in the MRNet architecture.

------------------------------------------------------------------------

## 6. Classification

The resulting 256-dimensional series representation was passed through:

``` text
Fully connected layer
        ↓
Sigmoid
        ↓
Probability from 0 to 1
```

The output represented the model's estimated probability that the
pathology was present.

The model was trained using **binary cross-entropy loss**.

Because the classes were imbalanced, the loss was weighted inversely
according to class prevalence so that uncommon positive cases had
greater influence during training.

------------------------------------------------------------------------

## 7. Separate Models for Plane and Diagnosis

MRNet did not use one neural network to predict everything
simultaneously.

It trained separate networks for each:

-   diagnosis
-   anatomical plane

There were three diagnoses:

``` text
Abnormality
ACL tear
Meniscal tear
```

and three planes:

``` text
Sagittal
Coronal
Axial
```

Therefore:

``` text
3 diagnoses × 3 planes = 9 MRNet models
```

For example:

``` text
Sagittal → ACL MRNet
Coronal  → ACL MRNet
Axial    → ACL MRNet
```

Each produced its own probability.

------------------------------------------------------------------------

## 8. Combining the Three MRI Planes

The predictions from the sagittal, coronal, and axial models were
combined using **logistic regression**.

For one diagnosis:

``` text
Sagittal probability ─┐
Coronal probability  ─┼→ Logistic regression → Final exam probability
Axial probability    ─┘
```

This allowed the final classifier to learn how informative each
anatomical plane was for a particular diagnosis.

The CNNs therefore learned image features, while logistic regression
acted as the final **multi-plane ensemble**.

------------------------------------------------------------------------

## 9. Training

Training followed the standard supervised deep-learning process:

``` text
MRI
 ↓
MRNet prediction
 ↓
Compare prediction with true label
 ↓
Binary cross-entropy loss
 ↓
Backpropagation
 ↓
Update neural-network parameters
```

Importantly, the AlexNet feature extractor was part of the trained MRNet
architecture rather than merely being used once to generate a static
dataset of features.

------------------------------------------------------------------------

## 10. Internal Validation Results

On the internal validation set, MRNet achieved approximately:

  Task                    AUC
  --------------- -----------
  Abnormality       **0.937**
  ACL tear          **0.965**
  Meniscal tear     **0.847**

ACL tear detection was the strongest of the three tasks.

Meniscal tear detection was substantially more difficult.

These results demonstrated that relatively simple slice aggregation
could still produce strong exam-level MRI classification.

------------------------------------------------------------------------

## 11. External Validation

The researchers also tested ACL detection using an external MRI dataset
from a different institution in Croatia.

Without retraining on that institution's data, the Stanford-trained
model achieved:

``` text
AUC = 0.824
```

When a model was trained using the external dataset, performance
increased to:

``` text
AUC = 0.911
```

This was important because it demonstrated both:

-   some ability to generalize across institutions and MRI acquisition
    differences
-   a noticeable performance penalty from **domain shift**

Domain shift remains a major issue in medical imaging.

------------------------------------------------------------------------

## 12. Comparison with Clinicians

The study included:

-   7 board-certified general radiologists
-   2 orthopedic surgeons

They interpreted MRI examinations both:

-   without MRNet assistance
-   with MRNet probabilities available

The study found that MRNet achieved performance comparable to general
radiologists for some tasks.

Providing model predictions also produced statistically significant
improvement in clinicians' **specificity for ACL tears**.

The intended role of MRNet was therefore not necessarily to replace
radiologists.

A more realistic use was:

``` text
MRI
 ↓
AI preliminary assessment
 ↓
Radiologist interpretation
 ↓
Final clinical decision
```

------------------------------------------------------------------------

## 13. Major Strengths of the MRNet Approach

### Handles variable numbers of slices

Max pooling allows an MRI series with 20 slices and one with 40 slices
to both become the same fixed-length representation.

### Uses transfer learning

The model begins with visual features learned from ImageNet rather than
learning every visual representation from scratch.

### Simple aggregation

MRNet avoids a complicated 3D CNN.

Every slice is processed using a 2D CNN, and information is combined
afterward.

This substantially reduces computational complexity.

### Multi-plane information

Sagittal, coronal, and axial MRI series provide different views of knee
anatomy.

MRNet allows each plane to develop specialized features before combining
predictions.

### Clinically interpretable output

The final output is a probability for a specific pathology rather than
an abstract feature representation.

------------------------------------------------------------------------

## 14. Limitations

The study had several important limitations.

### Relatively small dataset

1,370 MRI examinations is small compared with modern computer-vision
datasets.

### Limited diagnostic targets

The original model predicted only:

-   general abnormality
-   ACL tear
-   meniscal tear

It was not designed to classify the much larger range of abnormalities
that can appear on knee MRI.

### Labels derived from reports

Training labels were extracted from clinical reports rather than being
established through surgical confirmation.

### Single-institution training data

Most development data came from Stanford, which creates the possibility
that the model learns scanner, protocol, or population-specific
patterns.

### Simple slice aggregation

Max pooling asks whether a feature occurs strongly on **any** slice.

It does not explicitly model:

-   the order of slices
-   relationships between adjacent slices
-   full 3D anatomy

More modern approaches can potentially improve on this.

------------------------------------------------------------------------

## 15. Why MRNet Matters for Our RSNA Knee Project

The most valuable part of MRNet for our project is not necessarily
AlexNet itself.

The important architectural idea is:

``` text
Variable number of MRI slices
        ↓
Shared 2D CNN feature extractor
        ↓
Feature vector for every slice
        ↓
Aggregate across slices
        ↓
Series-level representation
        ↓
Classifier
```

This is directly applicable to a dataset containing variable-length
DICOM series.

However, we can modernize several components.

### Original MRNet

``` text
DICOM / MRI slices
      ↓
AlexNet
      ↓
256 features per slice
      ↓
Max pooling across slices
      ↓
256 series features
      ↓
Fully connected classifier
```

### Our Proposed Modernized Version

``` text
DICOM slices
      ↓
MRI preprocessing + augmentation
      ↓
Pretrained ResNet-18
      ↓
512 features per slice
      ↓
Series aggregation
      ↓
Series-level representation
      ↓
Multi-label classifier
      ↓
RSNA abnormality probabilities
```

The conceptual structure remains MRNet-style even though the CNN
backbone is changed.

------------------------------------------------------------------------

## 16. Why Use ResNet-18 Instead of AlexNet?

MRNet was published in 2018 and used an AlexNet-based feature extractor.

AlexNet was historically important, but ResNet introduced residual
connections that make deeper networks easier to optimize.

For our implementation, **ResNet-18 is a reasonable modern replacement
for AlexNet** because it:

-   provides stronger pretrained visual representations
-   remains relatively lightweight
-   produces a convenient 512-dimensional feature vector
-   is widely supported in PyTorch
-   is practical for experimentation before moving to larger backbones

Therefore, using ResNet does **not** mean we are abandoning the MRNet
idea.

We are retaining the important MRNet structure:

``` text
slice CNN → slice features → aggregation → series classifier
```

while replacing the older CNN backbone.

------------------------------------------------------------------------

## 17. MRNet vs. Our Planned Architecture

  -----------------------------------------------------------------------
  Component               Original MRNet          Planned RSNA Approach
  ----------------------- ----------------------- -----------------------
  Input                   Knee MRI series         Knee MRI DICOM series

  Slice model             AlexNet                 ResNet-18

  Pretraining             ImageNet                ImageNet

  Slice features          256                     512

  Variable slices         Supported               Supported

  Basic aggregation       Max pooling             Start with max/mean
                                                  pooling; evaluate
                                                  improvements

  Targets                 3                       Multiple RSNA
                                                  abnormalities

  Planes                  Sagittal, coronal,      Use available
                          axial                   series/plane metadata

  Final combination       Logistic regression     To be evaluated
                          across planes           

  Training framework      CNN classification      MRNet-style multi-label
                                                  pipeline
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 18. Core Lesson from MRNet

The central insight of MRNet is:

> **Treat an MRI examination as a collection of related 2D images, learn
> features from every slice, and then aggregate those features into a
> fixed-size representation that can be classified.**

The model does not need to convert the entire examination into one
enormous 3D image.

Instead:

``` text
slice
slice
slice
slice
  ↓
same CNN
  ↓
feature vectors
  ↓
aggregation
  ↓
one MRI representation
  ↓
prediction
```

That makes MRNet a useful architectural starting point for our RSNA knee
abnormality model even though we can replace AlexNet, improve
aggregation, and expand the number of predicted abnormalities.

------------------------------------------------------------------------

## Reference

Bien N, Rajpurkar P, Ball RL, Irvin J, Park A, Jones E, et al. (2018).
**Deep-learning-assisted diagnosis for knee magnetic resonance imaging:
Development and retrospective validation of MRNet.** *PLOS Medicine*,
15(11): e1002699.

DOI: 10.1371/journal.pmed.1002699
