# Baseline Models - Reorganized Structure

This directory contains the baseline statistical models for ontology completion, reorganized into a clear modular structure.

## Directory Structure

```
baseline/
├── unary/                    # Unary template prediction (UT)
│   ├── __init__.py
│   ├── unary_classify.py     # Main unary classification script
│   ├── unary_model.py        # Statistical model for unary classification
│   └── data_preprocessing_uni.py  # Unary data preprocessing
├── binary/                   # Binary template prediction (BT)
│   ├── __init__.py
│   ├── binary_predict.py     # Main binary prediction script
│   ├── binary_model.py       # Statistical model for binary prediction
│   └── data_preprocessing_bi.py  # Binary data preprocessing
├── shared/                   # Shared utilities
│   ├── __init__.py
│   ├── utils.py              # Common utility functions
│   ├── word_embedding.py     # Word embedding utilities
│   ├── student_t.py          # Statistical distribution functions
│   └── combined_features.py  # Combines results from both approaches
├── run_unary.py              # Wrapper script for unary classification
├── run_binary.py             # Wrapper script for binary prediction
└── run_combined.py           # Wrapper script for combined features
```

## Usage

### Running Individual Scripts

**Unary Classification:**
```bash
# Using wrapper script (recommended)
python run_unary.py

# Or directly
python unary/unary_classify.py
```

**Binary Prediction:**
```bash
# Using wrapper script (recommended)
python run_binary.py

# Or directly
python binary/binary_predict.py
```

**Combined Features:**
```bash
# Using wrapper script (recommended)
python run_combined.py

# Or directly
python shared/combined_features.py
```

### Module Imports

**For Unary Tasks:**
```python
from unary import unary_classify, unary_model, data_preprocessing_uni
```

**For Binary Tasks:**
```python
from binary import binary_predict, binary_model, data_preprocessing_bi
```

**For Shared Utilities:**
```python
from shared import utils, word_embedding, student_t, combined_features
```

## Model Components

### Unary Classification (`unary/`)
- **Purpose**: Predicts unary templates (UT) - single concept relationships
- **Key Files**:
  - `unary_classify.py`: Main execution script with 10-fold cross-validation
  - `unary_model.py`: Statistical model using Student-t distributions
  - `data_preprocessing_uni.py`: Handles unary template data preprocessing

### Binary Prediction (`binary/`)
- **Purpose**: Predicts binary templates (BT) - relationship between concept pairs
- **Key Files**:
  - `binary_predict.py`: Main execution script with 10-fold cross-validation
  - `binary_model.py`: Statistical model using Student-t distributions
  - `data_preprocessing_bi.py`: Handles binary template data preprocessing

### Shared Utilities (`shared/`)
- **Purpose**: Common functionality used by both unary and binary approaches
- **Key Files**:
  - `utils.py`: Evaluation metrics, rule finding, threshold selection
  - `word_embedding.py`: Word embedding processing using pre-trained vectors
  - `student_t.py`: Student-t distribution fitting and computation
  - `combined_features.py`: Combines results from both unary and binary models

## Features

### Supported Feature Types
1. **Word Embeddings**: Pre-trained word vectors (GoogleNews-vectors-negative300.bin.gz)
2. **Analogy Space Features**: Computed from word embeddings using analogy transformations

### Statistical Methods
- **Student-t Distribution**: Used for modeling feature distributions
- **Maximum Likelihood Estimation**: For parameter fitting
- **Bayesian Inference**: With likelihood ratios for classification
- **Threshold Selection**: Optimized for F1-score

### Evaluation Metrics
- **Precision**: Correctly predicted rules / Total predicted rules
- **Recall**: Correctly predicted rules / Total true rules
- **F1-Score**: Harmonic mean of precision and recall
- **10-fold Cross-validation**: For robust evaluation

## Dependencies

- gensim (for word embeddings)
- scikit-learn (for machine learning utilities)
- scipy (for statistical functions)
- pandas (for data manipulation)
- numpy (for numerical operations)

## Migration from Old Structure

The reorganization maintains backward compatibility through wrapper scripts. The main changes are:

1. **Modular Organization**: Related files are grouped together
2. **Clear Separation**: Unary and binary tasks are separated
3. **Shared Resources**: Common utilities are centralized
4. **Import Updates**: All import statements have been updated to use relative paths

## Benefits of Reorganization

1. **Better Organization**: Clear separation of concerns
2. **Easier Maintenance**: Related files are grouped together
3. **Consistent Structure**: Matches the GCN folder organization
4. **Improved Readability**: Logical file organization
5. **Future-Proof**: Easier to extend with new features

## File Dependencies

```
unary_classify.py
├── unary_model.py
├── data_preprocessing_uni.py
├── shared/student_t.py
└── shared/utils.py

binary_predict.py
├── binary_model.py
├── data_preprocessing_bi.py
├── shared/student_t.py
└── shared/utils.py

data_preprocessing_uni.py
├── shared/word_embedding.py
└── shared/utils.py

data_preprocessing_bi.py
├── shared/word_embedding.py
└── shared/utils.py
```

This structure provides a clean, maintainable, and extensible foundation for the baseline statistical models in ontology completion.
