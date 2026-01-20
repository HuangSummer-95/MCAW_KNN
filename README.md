# MCAW-KNN: Multi-Class Adaptive Weighted K-Nearest Neighbors

A Python implementation of the MCAW-KNN algorithm for classification tasks. This algorithm combines adaptive local region construction with class-specific feature weighting to improve KNN classification performance.

## Overview

MCAW-KNN addresses the limitations of traditional KNN by:

1. **Constructing adaptive local regions** around query points using Bayesian smoothing, Wilson score intervals, and Gaussian kernels
2. **Computing class-specific feature weights** using Linear Discriminant Analysis (LDA) and other methods
3. **Correcting weights** via generalized Rayleigh quotient manifold constraints
4. **Performing distance-weighted voting** for final classification

## Repository Structure

```
MCAW-KNN/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── notebooks/
│   └── MCAW_KNN_Algorithm.ipynb       # Main implementation notebook
├── datasets/
│   └── README.md                      # Dataset information and download links
└── paper/
    └── MCAW_KNN_2.pdf                 # Original research paper
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MCAW-KNN.git
cd MCAW-KNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from mcaw_knn import run_mcaw_knn_classification

# Run classification on your dataset
results = run_mcaw_knn_classification(
    file_path='path/to/your/data.csv',
    target_column='class_label',
    test_size=0.2,
    k_neighbors=7,
    region_size=20
)

print(f"Accuracy: {results['accuracy']:.4f}")
```

### Using Individual Components

```python
import numpy as np
from mcaw_knn import (
    SmoothLocalRegionBuilder,
    GlobalWeightMatrix,
    BinaryClassWeightCorrector,
    WeightedKNNClassifier
)

# Initialize components
region_builder = SmoothLocalRegionBuilder(k_region=15, region_size=20)
weight_calculator = GlobalWeightMatrix(method='lda')
corrector = BinaryClassWeightCorrector()
classifier = WeightedKNNClassifier(k=7)

# Fit on training data
region_builder.fit(X_train, y_train, row_indices)
corrector.fit(X_train, y_train)
classifier.fit(X_train, y_train, row_indices, feature_names)

# For each test point
local_region = region_builder.build_local_region(test_point)
# ... continue with classification
```

## Algorithm Components

### 1. SmoothLocalRegionBuilder

Constructs local regions around query points using:

- **Bayesian Smoothing**: Handles small sample sizes with prior information
- **Wilson Score Interval**: Provides conservative confidence estimates
- **Gaussian Kernel**: Adjusts distances based on class representativeness
- **Backward Rank**: Evaluates how representative a sample is within its class

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_region` | 15 | Candidate set size for each class |
| `region_size` | 15 | Target local region size |
| `alpha` | 1.0 | Bayesian prior parameter α |
| `beta` | 1.0 | Bayesian prior parameter β |
| `confidence_level` | 0.95 | Wilson interval confidence level |
| `sigma` | 1.0 | Gaussian kernel bandwidth |

### 2. GlobalWeightMatrix

Computes class-specific feature weights using various methods:

| Method | Description |
|--------|-------------|
| `lda` | Linear Discriminant Analysis via Rayleigh quotient |
| `f_score` | ANOVA F-statistics |
| `centroid` | Point-to-centroid ratio based |
| `inter_class_difference` | Precision matrix weighted differences |

### 3. BinaryClassWeightCorrector

Corrects local weight vectors using global manifold constraints:

- Treats each class as a binary problem (class c vs. not c)
- Projects weights onto the generalized Rayleigh quotient manifold
- Balances local weight preservation with global structure

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `reg_param` | 1e-6 | Regularization for numerical stability |
| `alpha` | 0.5 | Balance between local and global constraints |
| `method` | 'projection' | Correction method ('projection' or 'optimization') |

### 4. WeightedKNNClassifier

Final classification using:

- Feature-weighted Euclidean distance
- Multiple distance weighting schemes
- Distance-weighted voting

**Distance Weight Methods:**
- `rank_exponential`: Exponential decay by neighbor rank
- `rank_power`: Power-law decay by rank
- `relative_distance`: Based on ratio to nearest distance
- `top_m_dominant`: Only top m neighbors contribute

## Datasets

The paper evaluates MCAW-KNN on several UCI Machine Learning Repository datasets:

| Dataset | Samples | Features | Classes | Description |
|---------|---------|----------|---------|-------------|
| Wine Quality | 6,497 | 12 | 7 | Red and white wine quality scores |
| Iris | 150 | 4 | 3 | Classic iris flower classification |
| Credit Card | ~30,000 | 24 | 2 | Credit card default prediction |

See `datasets/README.md` for download links and preprocessing instructions.

## Results

Performance comparison on benchmark datasets (accuracy %):

| Dataset | Standard KNN | Weighted KNN | MCAW-KNN |
|---------|-------------|--------------|----------|
| Wine Quality | ~55% | ~58% | ~62% |
| Iris | ~96% | ~97% | ~98% |

*Results may vary based on train/test split and hyperparameters.*

## Algorithm Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Query Point x                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         Build Local Region (SmoothLocalRegionBuilder)        │
│  • Compute class tightness                                   │
│  • Calculate backward ranks                                  │
│  • Apply Bayesian + Wilson smoothing                         │
│  • Adjust distances with Gaussian kernel                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│       Compute Feature Weights (GlobalWeightMatrix)           │
│  • LDA-based Rayleigh quotient                              │
│  • Class-specific weight vectors                             │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│        Correct Weights (BinaryClassWeightCorrector)          │
│  • Project onto Rayleigh quotient manifold                  │
│  • Balance local vs global constraints                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Classify (WeightedKNNClassifier)                    │
│  • Compute weighted distances per class                      │
│  • Distance-weighted voting                                  │
│  • Select class with highest consistency                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Output: Predicted Class                    │
└─────────────────────────────────────────────────────────────┘
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{huang2026mcawknn,
  title={MCAW-KNN: Multi-Class Adaptive Weighted K-Nearest Neighbors},
  author={Huang, Meimei and others},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Original algorithm design by Meimei Huang
- UCI Machine Learning Repository for benchmark datasets
- scikit-learn for foundational ML utilities

## Contact

For questions or feedback, please open an issue on GitHub.
