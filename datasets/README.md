# Datasets

This directory contains datasets used to evaluate the MCAW-KNN algorithm.

## Downloading Datasets

The datasets used in the paper are available from the UCI Machine Learning Repository:

### 1. Wine Quality Dataset

**Description:** Two datasets related to red and white variants of Portuguese "Vinho Verde" wine. Contains physicochemical properties and quality ratings.

**Download:** [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

```bash
# Download directly
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
```

**Features:**
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- type (red/white)

**Target:** quality (score between 3-9)

**Preprocessing:**
```python
import pandas as pd

# Load and combine datasets
red = pd.read_csv('winequality-red.csv', sep=';')
red['type'] = 'red'
white = pd.read_csv('winequality-white.csv', sep=';')
white['type'] = 'white'

# Combine
wine = pd.concat([red, white], ignore_index=True)
wine.to_csv('winequalityN.csv', index=False)
```

---

### 2. Iris Dataset

**Description:** Classic dataset containing measurements of iris flowers from three species.

**Download:** [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

**Features:**
- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

**Target:** species (setosa, versicolor, virginica)

---

### 3. Credit Card Default Dataset

**Description:** Default payments of credit card clients in Taiwan.

**Download:** [Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

**Features:**
- Credit limit
- Gender, Education, Marital status, Age
- Payment history (6 months)
- Bill amounts (6 months)
- Payment amounts (6 months)

**Target:** default payment next month (1=yes, 0=no)

---

## File Placement

After downloading, place the CSV files in this directory:

```
datasets/
├── README.md
├── winequalityN.csv
├── iris.csv
└── credit_card.csv
```

## Data Format Requirements

All datasets should be in CSV format with:
- First row containing column headers
- Target variable in a column (specify name when running)
- Numeric features (categorical features will be encoded)
- No missing values (or they will be dropped)

## Adding Your Own Datasets

To use MCAW-KNN with your own dataset:

1. Ensure data is in CSV format
2. Place in the `datasets/` directory
3. Update the `file_path` and `target_column` parameters when calling the algorithm:

```python
results = run_mcaw_knn_classification(
    file_path='datasets/your_data.csv',
    target_column='your_target_column',
    test_size=0.2,
    k_neighbors=7,
    region_size=20
)
```
