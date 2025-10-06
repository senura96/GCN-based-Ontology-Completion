# GCN-based Ontology Completion

This repository contains implementations of Graph Convolutional Networks (GCNs) for ontology completion tasks, including both baseline statistical methods and advanced neural network approaches.

## Project Structure

```
├── code/
│   ├── baseline/           # Traditional statistical approaches
│   └── gcn/               # Graph Neural Network implementations
│       ├── unary_classify/    # Unary template prediction
│       └── binary_predict/    # Binary template prediction
├── dataset/
│   ├── contor_datasets/       # Main dataset with train/dev/test splits
│   ├── contor_untyped/        # Untyped version of main dataset
│   ├── contor_datasets_split_types/  # Specialized test sets by rule type
│   └── transport/             # 10-fold cross-validation datasets
```

## Datasets

### 1. contor_datasets (Main Dataset)
**Location**: `dataset/contor_datasets/transport/`

**Structure**:
- **train.json**: 1,903 samples
- **dev.json**: 482 samples  
- **test.json**: 261 samples
- **Total**: 2,646 samples

**Characteristics**:
- Standard train/dev/test split for machine learning evaluation
- Label distribution: 1,615 negative (label: 0) vs 289 positive (label: 1) in training set
- Contains both positive rules (head=concept) and negative rules (head=owl.Bottom)
- Purpose: Main dataset for training and evaluating ontology completion models

**Data Format**:
```json
{
  "v_sub_concept": "tunnel",
  "v_super_concept": "stationary artifact", 
  "label": 1,
  "rule": "body=Mid-level-ontology.Tunnel, head=SUMO.StationaryArtifact"
}
```

### 2. contor_untyped (Untyped Version)
**Location**: `dataset/contor_untyped/transport/`

**Structure**:
- **train.json**: 1,903 samples
- **dev.json**: 482 samples
- **test.json**: 261 samples
- **Total**: 2,646 samples

**Characteristics**:
- Identical content to `contor_datasets`
- Same label distribution and rule patterns
- Purpose: Used for experiments where type information is removed or ignored
- Use case: Backup or alternative version of the main dataset

### 3. contor_datasets_split_types (Specialized Test Sets)
**Location**: `dataset/contor_datasets_split_types/untyped/transport/`

**Structure**:
- **test_basic.json**: 170 samples (81 negative, 89 positive)
- **test_intersection.json**: 92 samples (92 negative, 0 positive)
- **Total**: 262 samples

**Characteristics**:
- Specialized test splits based on rule complexity
- **test_basic**: Contains simpler, more straightforward rules
- **test_intersection**: Contains only negative examples (head=owl.Bottom) - intersection/disjointness rules
- No training data - only test sets
- Purpose: Evaluates model performance on different types of ontological relationships

**Rule Pattern Examples**:

**test_basic** (Simple subsumption):
```json
{
  "v_sub_concept": "head end car",
  "v_super_concept": "railcar",
  "label": 1,
  "rule": "body=transport.HeadEndCar, head=transport.Railcar"
}
```

**test_intersection** (Disjointness rules):
```json
{
  "v_sub_concept": "dredger",
  "v_super_concept": "motor hopper", 
  "label": 0,
  "rule": "body=transport.Dredger, transport.MotorHopper, head=owl.Bottom"
}
```

## Model Implementations

### Baseline Models (Statistical Approaches)
**Location**: `code/baseline/`

**Components**:
- **Unary Classification** (`unary_classify.py`): Predicts unary templates (UT)
- **Binary Prediction** (`binary_predict.py`): Predicts binary templates (BT)  
- **Combined Features** (`combined_features.py`): Combines results from both approaches

**Architecture**:
- Statistical approach using probability distributions
- Feature types: Word embeddings and analogy space features
- Training method: Maximum likelihood estimation with Gaussian/Student-t distributions
- Loss function: Log-likelihood optimization

**Key Algorithms**:
- Computes PDFs for node features, pair features, and relation templates
- Uses Bayesian inference with likelihood ratios
- Applies threshold selection for binary classification

### GCN Unary Classification
**Location**: `code/gcn/unary_classify/`

**Architecture**:
- Graph-based approach using DGL (Deep Graph Library)
- RGCN layers with relation-specific convolutions
- Multi-label classification for unary templates
- Feature types: Word embeddings, analogy space features, or concatenated features
- Loss function: BCEWithLogitsLoss + similarity loss (σ·sim_loss_in + β·sim_loss_out)

**Key Features**:
- Similarity constraints: Nodes with similar incoming/outgoing relations have similar embeddings
- Multi-layer architecture: Input → Hidden → Output layers
- Regularization: Dropout, gradient clipping, L2 normalization
- Validation: Uses threshold selection for optimal F1 score

### GCN Binary Prediction
**Location**: `code/gcn/binary_predict/`

**Architecture**:
- Graph-based approach with RGCN backbone
- DistMult scoring function: `score = Σ(s * r * o)` where s, r, o are subject, relation, object embeddings
- Binary classification for relation existence
- Training strategy: Negative sampling with graph batching
- Loss function: Binary cross-entropy + L2 regularization

**Key Features**:
- Efficient training: Uses negative sampling and graph batching
- Relation embeddings: Learnable relation-specific parameters
- Regularization: L2 regularization on embeddings and relation weights
- Early stopping: Based on validation F1 score
- Model checkpointing: Saves best model during training

## Feature Types

All models support two main feature types:
1. **Word Embeddings**: Pre-trained word vectors (GoogleNews-vectors-negative300.bin.gz)
2. **Analogy Space Features**: Computed from word embeddings using analogy transformations

**Additional GCN Variations**:
- **Concatenated Features**: Combines word embeddings and analogy features
- **Combined PCA**: Uses PCA to combine different feature types
- **All Data Used**: Tests on complete datasets without train/test splits

## Evaluation

**Metrics**:
- **Precision**: Correctly predicted rules / Total predicted rules
- **Recall**: Correctly predicted rules / Total true rules  
- **F1-Score**: Harmonic mean of precision and recall

**Cross-validation**: 10-fold cross-validation for robust evaluation

**Evaluation Strategy**:
The split_types dataset provides hierarchical evaluation:
1. **Basic reasoning**: Can the model handle simple "is-a" relationships?
2. **Intersection reasoning**: Can the model identify when concepts are disjoint?

## Dependencies

### Baseline Models
- gensim
- scikit-learn
- scipy
- pandas
- numpy

### GCN Models
- dgl
- pytorch
- pandas
- numpy

## Usage

### Baseline Models
```bash
# Unary classification
python unary_classify.py

# Binary prediction  
python binary_predict.py

# Combined features
python combined_features.py
```

### GCN Models
```bash
# Unary classification with GCN
python unary_classify.py -d wine -f embedding

# Binary prediction with GCN
python binary_predict.py -d wine -f analogy

# Combined features
python combined_pca_and_we.py
```

## Dataset Evaluation Framework

The three datasets together provide a comprehensive evaluation framework:

1. **contor_datasets**: Primary dataset for general ontology completion performance
2. **contor_untyped**: Alternative version for type-agnostic experiments  
3. **contor_datasets_split_types**: Specialized evaluation for different reasoning types

This allows researchers to understand which types of ontological relationships their models handle well and where they struggle.

## File Structure Details

### Dataset Files
- **JSON format**: Each line contains a JSON object with concept pairs, labels, and rules
- **Labels**: 1 for positive relationships, 0 for negative/disjoint relationships
- **Rules**: Formal logical rules in the format `body=concept1, head=concept2` or `head=owl.Bottom` for disjointness

### Model Files
- **Baseline**: Statistical models with probability distributions
- **GCN**: Neural network models with graph convolutions
- **Utilities**: Data preprocessing, evaluation metrics, and helper functions

## Citation

If you use this code or datasets, please cite the original paper:

```bibtex
@article{ontology_completion_gcn,
  title={GCN-based Ontology Completion},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]}
}
```
