# SSC-Unal: Spatial Structure Complexity-Aware Uncertainty Alignment for Active Learning Individual Tree Crown Detection

Official repository for the paper **"SSC-Unal: Spatial Structure Complexity-Aware Uncertainty Alignment for Active Learning Individual Tree Crown Detection"**.

## Overview

This work proposes a spatial-structure-complexity-aware uncertainty alignment strategy (SSC-Unal) for active learning in individual tree crown detection (ITCD). The method leverages structural priors from aerial/satellite imagery to guide sample selection, improving labeling efficiency for forest inventory tasks.

### Datasets

| Dataset | Version | Samples | Classes | Annotation Format |
|---------|---------|---------|---------|-------------------|
| Spruce-Bark-Beetle | v1.0 | aerial forest images | 5 tree species | VOC XML → COCO |
| TreeAI | v1.2 | 640×640 RGB patches | 12 → top-10 | YOLO txt → COCO |

### Active Learning Split

Each dataset is preprocessed into four subsets:

| Split | Default Ratio | Purpose |
|-------|---------------|---------|
| `init_train` | 5% | Initial labeled training set |
| `val` | 10% | Validation set |
| `test` | 10% | Held-out test set |
| `unlabeled` | 75% | Unlabeled pool (GT retained for AL simulation) |

## Repository Structure

