# AUGMENTATION-SPEC — Data Augmentation Specification

## 1. Purpose
Simulate variation in lighting and skin appearance to improve robustness and reduce the
performance gap across skin tones. Augmentation is applied **only to the training set**
in the `augmented` experiment.

## 2. Baseline (no augmentation)
Training transform = preprocessing only (resize, normalize). See [DATA-SPEC.md](DATA-SPEC.md).

## 3. Augmented Pipeline (training only)
Applied before normalization, in order:

| Transform | Parameters |
|-----------|------------|
| RandomHorizontalFlip | p=0.5 |
| RandomRotation | ±20° |
| ColorJitter — brightness | 0.2 |
| ColorJitter — contrast | 0.2 |
| ColorJitter — saturation | 0.2 |
| ColorJitter — hue | 0.1 |

Then resize to 224×224 and normalize with ImageNet stats.

## 4. torchvision Reference
```python
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

## 5. Validation / Test
No augmentation — preprocessing only, for both experiments.

## 6. Rationale
HAM10000 lacks skin-tone labels and is lighter-skinned on average. Photometric jitter
approximates darker-skin appearance in the absence of real diverse-tone training data.
This limitation is documented in [LIMITATIONS-SPEC.md](LIMITATIONS-SPEC.md).

## 7. Output
Generate an "Augmentation Examples" figure (one lesion under several augmentations) —
see [FIGURE-SPEC.md](FIGURE-SPEC.md).
