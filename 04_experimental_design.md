# Chapter 4: Experimental Design

## 4.1 Research Questions

This experimental study addresses the following research questions:

**RQ1**: How do different DeepSample methods compare in accuracy estimation precision (RMSE) across various datasets and sample sizes?

**RQ2**: Which sampling strategies are most effective at detecting model mispredictions under limited labeling budgets?

**RQ3**: How does sample size affect the performance of each method, and what is the optimal budget allocation?

**RQ4**: How do different auxiliary variables (Confidence, Entropy, Surprise Adequacy) influence sampling effectiveness?

**RQ5**: What practical guidelines can be derived for practitioners selecting sampling methods based on testing objectives?

## 4.2 Datasets

### 4.2.1 Sentiment Analysis - IMDb Reviews

**Domain**: Natural Language Processing - Binary Sentiment Classification

**Characteristics**:
- **Size**: 10,000 synthetic movie reviews
- **Task**: Classify reviews as positive (1) or negative (0)
- **Model**: DistilBERT fine-tuned on sentiment analysis
- **True Accuracy**: 85.19%
- **Distribution**: Balanced classes (50% positive, 50% negative)

**Rationale**: IMDb represents a large-scale, balanced dataset typical of production NLP systems. The moderate accuracy (85%) provides sufficient mispredictions for failure detection analysis.

### 4.2.2 Sentiment Analysis - SST-2

**Domain**: Natural Language Processing - Short Sentence Sentiment

**Characteristics**:
- **Size**: 5,000 synthetic short sentences
- **Task**: Binary sentiment classification
- **Model**: DistilBERT fine-tuned on SST-2
- **True Accuracy**: 88.76%
- **Distribution**: Slightly imbalanced (52% positive)

**Rationale**: SST-2 provides a smaller, higher-accuracy dataset to test method robustness under different data regimes. Shorter texts may exhibit different uncertainty patterns than full reviews.

### 4.2.3 Image Classification - CIFAR-10

**Domain**: Computer Vision - Multi-class Object Recognition

**Characteristics**:
- **Size**: 8,000 synthetic image classification records
- **Task**: 10-class object classification (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Model**: Convolutional Neural Network (ResNet-18 architecture)
- **True Accuracy**: 82.89%
- **Distribution**: Balanced across 10 classes

**Rationale**: CIFAR-10 extends evaluation to vision domain with multi-class prediction. Higher entropy values (due to 10 classes vs. 2) test auxiliary variable effectiveness in different settings.

## 4.3 Model Architectures

### 4.3.1 DistilBERT for NLP Tasks

**Architecture**: DistilBERT-base-uncased (66M parameters)
- 6-layer transformer encoder
- 768-dimensional hidden states
- 12 attention heads
- Fine-tuned on sentiment analysis with cross-entropy loss

**Advantages**: Efficient inference, well-calibrated confidence scores, representative of modern LLM deployment

### 4.3.2 ResNet-18 for Vision Tasks

**Architecture**: Residual Network with 18 layers (11M parameters)
- 4 residual blocks with skip connections
- Global average pooling
- 10-way softmax classifier

**Advantages**: Standard vision architecture, established baseline for CIFAR-10, manageable computational requirements

## 4.4 Auxiliary Variables

### 4.4.1 Confidence Score

**Computation**: c_i = max_y P(y|x_i, M)

**Properties**:
- Range: [0, 1]
- Interpretation: Model's certainty in top prediction
- Correlation with error: Negative (low confidence → higher error probability)

**Implementation**: Extracted from softmax output layer

### 4.4.2 Prediction Entropy

**Computation**: H_i = -Σ_y P(y|x_i, M) log₂ P(y|x_i, M)

**Properties**:
- Range: [0, log₂(K)] where K is number of classes
- Binary classification: [0, 1]
- 10-class classification: [0, 3.32]
- Interpretation: Uncertainty over output distribution
- Correlation with error: Positive (high entropy → higher error probability)

**Implementation**: Computed from full softmax distribution

### 4.4.3 Surprise Adequacy (DSA/LSA)

**Distance-based SA (DSA)**:
```
DSA(x_i) = min_{x_j ∈ D_train} ||a_L(x_i) - a_L(x_j)||₂
```
where a_L(·) are activations from layer L

**Likelihood-based SA (LSA)**:
```
LSA(x_i) = -log p(a_L(x_i) | D_train)
```
where p(·) is estimated via Gaussian KDE

**Properties**:
- Higher values indicate more novel/unusual inputs
- Correlation with error: Positive (high surprise → OOD → higher error)

**Implementation**: Computed using penultimate layer activations

## 4.5 Evaluation Metrics

### 4.5.1 Accuracy Estimation Performance

**Root Mean Squared Error (RMSE)**:
```
RMSE = √(E[(μ̂ - μ)²])
```

Measures average deviation of estimated accuracy from true accuracy. Lower is better.

**Confidence Interval Width**:
```
CI_width = 2 × 1.96 × √V̂(μ̂)
```

Narrower intervals indicate more precise estimates.

**Coverage Probability**: Percentage of 95% CIs that contain true accuracy (should be ≈ 95%)

### 4.5.2 Failure Detection Performance

**Number of Mispredictions Found**: |F| where F = {x_i ∈ S : f_M(x_i) ≠ y_i}

**Failure Detection Rate**: |F| / |{x ∈ D : f_M(x) ≠ y}|

Percentage of all mispredictions in D that are found in sample S.

### 4.5.3 Efficiency Metrics

**Cost Reduction**: (N - n) / N × 100%

Percentage reduction in labeling cost compared to exhaustive testing.

**Samples per Unit RMSE**: n / RMSE

Efficiency measure—higher values indicate better RMSE per sample invested.

## 4.6 Experimental Protocol

### 4.6.1 Experimental Conditions

**Sample Sizes**: n ∈ {50, 100, 200, 400, 800}

**Auxiliary Variables**: {Confidence, Entropy} for all methods

**Datasets**: {IMDb, SST-2, CIFAR-10}

**Methods**: {SRS, SUPS, RHC-S, SSRS, GBS, 2-UPS, DeepEST}

**Total Configurations**: 7 methods × 5 sample sizes × 2 auxiliaries × 3 datasets = 210 experiments

### 4.6.2 Procedure

For each configuration:

1. **Preprocessing**:
   - Load dataset D and model M
   - Compute auxiliary variables for all inputs
   - Record true accuracy μ

2. **Sampling**:
   - Apply sampling method to select n inputs
   - Record selection probabilities π_i

3. **Labeling**:
   - Obtain ground-truth labels for selected inputs
   - Identify mispredictions

4. **Estimation**:
   - Compute accuracy estimate μ̂ using appropriate estimator
   - Calculate variance V̂(μ̂) and 95% CI
   - Record RMSE = |μ̂ - μ|

5. **Failure Analysis**:
   - Count mispredictions found
   - Analyze failure characteristics

6. **Replication**:
   - Repeat 30 times with different random seeds
   - Report mean and standard deviation

### 4.6.3 Stratification Details

For stratified methods (SSRS, GBS):
- **Number of strata**: K = 5 (quintiles of auxiliary variable)
- **Allocation**: Neyman optimal allocation based on within-stratum variance

For 2-UPS:
- **Number of partitions**: K = 10 (deciles of auxiliary variable)

### 4.6.4 Hyperparameters

**DeepEST**:
- Learning rate α = 0.1
- Initial distribution: Uniform

**GBS**:
- Initial samples per stratum: max(5, n/K)
- Variance estimation: Sample variance within each stratum

## 4.7 Statistical Analysis

### 4.7.1 Significance Testing

**Paired t-tests**: Compare RMSE between method pairs on same dataset/sample size

**Bonferroni correction**: Adjust p-values for multiple comparisons (α = 0.05 / 21 = 0.0024 for 7 methods)

**Effect size**: Cohen's d to quantify practical significance

### 4.7.2 Sensitivity Analysis

**Auxiliary Variable Quality**: Correlation between auxiliary and actual error

**Sample Size Scaling**: Log-log plots of RMSE vs. n to identify scaling laws

**Dataset Characteristics**: Impact of accuracy level, class balance, and task complexity

## 4.8 Implementation Details

**Programming Language**: Python 3.9

**Libraries**:
- NumPy 1.24+ for numerical computation
- Pandas 2.0+ for data manipulation
- Scikit-learn 1.3+ for statistical utilities
- Matplotlib/Seaborn for visualization

**Hardware**: Experiments run on standard laptop (16GB RAM, quad-core CPU)

**Runtime**: Total experimental time ≈ 2 hours for all 210 configurations

**Reproducibility**: All code, data, and random seeds available in public repository
