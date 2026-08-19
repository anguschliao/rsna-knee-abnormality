## 1. CNN

A **Convolutional Neural Network (CNN)** is a type of neural network
designed to recognize visual patterns in images.

A CNN learns increasingly complicated patterns as an image moves through
the network:

``` text
Knee MRI
   ↓
find simple edges and lines
   ↓
find shapes and textures
   ↓
find larger anatomical patterns
   ↓
combine everything
   ↓
"Does this look abnormal?"
   ↓
87% abnormal
```

Early layers tend to learn simple visual features such as:

-   edges
-   lines
-   brightness changes
-   simple textures

Deeper layers combine these into more complicated patterns.

A useful analogy is recognizing a face:

> First notice lines → then eyes and nose → then facial structure →
> eventually recognize the person.

The important idea is that the CNN **learns which visual patterns are
useful from the training images**.

------------------------------------------------------------------------

## 2. Why a Traditional CNN Is Not Ideal for Knee MRI

There are two major problems.

### Problem 1: Training From Scratch Requires a Lot of Data

A CNN created from scratch initially knows nothing about images.

``` text
Random CNN
    ↓
"What's an edge?"
"What's a shape?"
"What's a texture?"
    ↓
must learn everything
from the knee dataset
```

With millions of training images, this can work very well.

With a much smaller specialized medical dataset, however, the model has
less data from which to learn all of these concepts.

This increases the risk that the model learns peculiarities of the
training dataset rather than visual patterns that generalize to new
patients.

------------------------------------------------------------------------

### Problem 2: A Knee MRI Examination Is Not One Image

A knee MRI consists of a **series of slices** through the knee.

For example:

``` text
        Knee MRI

slice 1   ─────────
slice 2   ─────────
slice 3   ─────────
slice 4   ─────────
...
slice 30  ─────────
```

An abnormality may only be clearly visible on a few of those slices.

For example:

``` text
slice 14 → nothing obvious
slice 15 → slightly suspicious
slice 16 → suspicious
slice 17 → strong abnormal feature
slice 18 → suspicious
slice 19 → nothing obvious
```

If we simply train:

``` text
one slice → CNN → abnormal / normal
```

we may be asking the wrong question.

The abnormality label generally applies to the **MRI examination or
study**, not necessarily to every individual slice.

Therefore, a completely normal-looking slice can still come from an
abnormal knee.

------------------------------------------------------------------------

## 3. Why a Pretrained ResNet Is Useful

**ResNet (Residual Network)** is a type of CNN.

Rather than building a ResNet with completely random weights, we can
begin with a model that has already been trained on a very large image
dataset.

The basic idea is:

### CNN Trained From Scratch

``` text
training data
     ↓
learn basic visual patterns
     ↓
learn more complicated image features
     ↓
learn knee-specific patterns
     ↓
learn abnormalities
```

### Pretrained ResNet

``` text
already understands
general visual patterns
       ↓
knee MRI training data
       ↓
adapt those features to MRI
       ↓
learn knee abnormalities
```

This process is called **transfer learning**.

The knee MRI dataset can then be used to adapt those learned features to
the medical imaging problem.

------------------------------------------------------------------------

## 4. ResNet's Skip Connections

ResNet also introduced **residual or skip connections**.

A traditional deep network passes information sequentially:

``` text
A → B → C → D → E → F
```

ResNet adds shortcuts that allow information to bypass some layers:

``` text
     ┌───────────────┐
     │               ↓
A → B → C → D → E → F
         │       ↑
         └───────┘
```

Mathematically, a residual block can be thought of as:

``` text
output = learned transformation + original input
```

These shortcuts make very deep CNNs easier to train.

For this knee MRI project, however, the most important practical
advantage is that **a pretrained ResNet can be used as a powerful image
feature extractor**.

------------------------------------------------------------------------

## 5. Using ResNet as a Feature Extractor

Instead of asking ResNet to immediately decide whether an individual
slice represents an abnormal knee, we can use it to describe the visual
information in each slice.

For example:

``` text
MRI slice
    ↓
Pretrained ResNet
    ↓
learned feature representation
```

A ResNet-50 can produce a feature vector representing the visual
characteristics of the image.

Conceptually, these features might respond to things such as:

-   shapes
-   boundaries
-   textures
-   tissue patterns
-   brightness patterns
-   anatomical structures

The model does not literally create variables with these names. It
learns numerical representations that are useful for distinguishing
visual patterns.

------------------------------------------------------------------------

# 6. The MRNet-Style Approach

The MRNet-style approach addresses the fact that an MRI contains many
slices.

Instead of making a final diagnosis from each individual image, the same
CNN examines **every slice in the MRI series**.

For example:

``` text
slice 1  → ResNet → features
slice 2  → ResNet → features
slice 3  → ResNet → features
...
slice 30 → ResNet → features
```

The result is a collection of feature representations describing the
entire MRI series.

These features are then **aggregated across slices** before the final
classification is made.

A simplified architecture looks like:

``` text
             COMPLETE MRI

slice 1  ─→ ResNet ─┐
slice 2  ─→ ResNet ─┤
slice 3  ─→ ResNet ─┤
slice 4  ─→ ResNet ─┤
   ...               ├→ aggregate features
slice 17 ─→ ResNet ─┤       ↓
   ...               │    classifier
slice 30 ─→ ResNet ─┘       ↓
                         ABNORMAL
```

The important distinction is:

``` text
Individual slice
      ↓
    ResNet
      ↓
features
```

rather than:

``` text
Individual slice
      ↓
    ResNet
      ↓
final diagnosis for the entire knee
```

The final diagnosis is made **after information from the slices has been
combined**.

------------------------------------------------------------------------

## 7. Max Pooling Across MRI Slices

The original MRNet approach used **max pooling** to aggregate
information across slices.

In simplified terms, max pooling asks:

> What was the strongest evidence for each learned feature anywhere in
> this MRI series?

Imagine an abnormality is most visible on slice 17:

``` text
slice 14 → weak signal
slice 15 → weak signal
slice 16 → moderate signal
slice 17 → VERY STRONG SIGNAL
slice 18 → moderate signal
slice 19 → weak signal
```

Max pooling allows the strong feature detected on slice 17 to survive
when information from all of the slices is combined.

Conceptually:

``` text
30 MRI slices
      ↓
ResNet examines every slice
      ↓
features from every slice
      ↓
max pooling
      ↓
strongest evidence found anywhere
      ↓
classifier
      ↓
probability of abnormality
```

This is particularly useful when an abnormality is only visible on a
small number of slices.

------------------------------------------------------------------------

# 8. Putting Everything Together

The complete idea is:

``` text
                 KNEE MRI SERIES
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
     slice 1          slice 2         slice N
        ↓               ↓               ↓
     ResNet           ResNet           ResNet
        ↓               ↓               ↓
    features          features        features
        └───────────────┼───────────────┘
                        ↓
               aggregate across slices
                  (e.g. max pooling)
                        ↓
                    classifier
                        ↓
             abnormality probability
```

This combines three important ideas:

1.  **CNN:** learns visual patterns from images.
2.  **Pretrained ResNet:** starts with useful visual knowledge rather
    than learning everything from scratch.
3.  **MRNet-style aggregation:** examines the entire MRI series before
    making the study-level prediction.

For knee MRI abnormality detection, this provides a strong and
understandable baseline before experimenting with more advanced
approaches such as attention pooling, Transformers, 3D CNNs, or
ensembles.
