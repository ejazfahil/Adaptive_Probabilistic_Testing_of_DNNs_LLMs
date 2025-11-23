# Chapter 2: Background and Related Work

## 2.1 Deep Neural Network Testing Challenges

### 2.1.1 The Evaluation Cost Problem

Testing DNNs and LLMs presents unique challenges compared to traditional software. Unlike conventional programs with deterministic behavior, neural networks exhibit probabilistic outputs influenced by training data distribution, architectural choices, and optimization dynamics. Comprehensive evaluation requires:

- **Large-scale datasets**: Thousands to millions of labeled examples
- **Domain expertise**: Specialized annotators for medical, legal, or technical domains
- **Continuous re-evaluation**: Each model update necessitates retesting

**Cost Analysis**: Consider a medical imaging classifier requiring radiologist annotations at $50/image. Testing on 10,000 images costs $500,000—prohibitive for iterative development.

### 2.1.2 Traditional Evaluation Paradigms

**Holdout Testing**: The standard approach partitions data into training, validation, and test sets. While simple, it requires pre-labeling entire test sets and provides no mechanism for cost reduction.

**Cross-Validation**: K-fold CV improves estimate reliability but multiplies labeling costs by K, making it impractical for large-scale models.

**Benchmark Datasets**: Public benchmarks (ImageNet, GLUE, SuperGLUE) enable standardized comparison but suffer from:
- **Data contamination**: Models may have seen test data during pre-training
- **Saturation**: Top models achieve near-perfect scores, reducing discriminative power
- **Distribution shift**: Benchmark performance may not reflect operational accuracy

## 2.2 Probabilistic Sampling Theory Foundations

### 2.2.1 Survey Sampling Principles

DeepSample builds on classical survey sampling theory developed for population estimation. Key concepts include:

**Sampling Frame**: The complete set of units from which samples are drawn (analogous to the operational input distribution)

**Inclusion Probability**: π_i = P(unit i is selected). Unequal probabilities enable targeted sampling of high-value units.

**Horvitz-Thompson Estimator**: For estimating population totals with unequal probability sampling:

```
Ŷ = Σ (y_i / π_i)
```

where y_i is the observed value for sampled unit i. This estimator is unbiased regardless of selection probabilities.

**Variance Estimation**: The Horvitz-Thompson variance estimator accounts for pairwise inclusion probabilities:

```
V(Ŷ) = ΣΣ (π_ij - π_i π_j) × (y_i/π_i - y_j/π_j)²
```

### 2.2.2 Stratified Sampling

Partitioning the population into homogeneous strata and sampling within each stratum reduces variance. **Neyman allocation** optimizes sample allocation:

```
n_h ∝ N_h × σ_h
```

where n_h is the sample size for stratum h, N_h is the stratum size, and σ_h is the within-stratum standard deviation.

### 2.2.3 Adaptive Sampling

Modern adaptive methods adjust sampling probabilities based on observed data. **Sequential sampling** continues until a precision target is met, while **two-phase sampling** uses cheap auxiliary information to guide expensive measurements.

## 2.3 State-of-the-Art Evaluation Methods (2024)

### 2.3.1 SubLIME: Adaptive Benchmark Sampling

**SubLIME** (Less Is More for Evaluation) addresses the computational burden of evaluating LLMs on extensive benchmarks. Key innovations:

- **Clustering-based sampling**: Groups similar test items and samples representatives
- **Quality-based sampling**: Prioritizes high-quality, discriminative questions
- **Difficulty-based sampling**: Ensures coverage across difficulty levels

**Results**: SubLIME achieves accurate model rankings on MMLU with just 1% of samples (150 questions vs. 15,000), demonstrating 99x cost reduction.

**Limitations**: Designed for benchmark subset selection rather than operational testing; focuses on ranking preservation rather than unbiased accuracy estimation.

### 2.3.2 FLUID BENCHMARKING: Psychometric Adaptive Testing

**FLUID** applies Item Response Theory (IRT) from educational testing to LLM evaluation:

**Item Response Model**: Models the probability of correct response as a function of model ability θ and item difficulty b:

```
P(correct | θ, b) = 1 / (1 + exp(-(θ - b)))
```

**Adaptive Selection**: Chooses next test item to maximize information gain about model ability, similar to computerized adaptive testing (CAT) in standardized exams.

**Results**: 50x reduction on MMLU while maintaining higher validity than static benchmarks.

**Advantages**: Addresses benchmark saturation and data contamination through dynamic item selection.

### 2.3.3 Uncertainty Quantification Methods

Recent UQ advances provide robust auxiliary variables for DeepSample:

**Deep Ensembles (DE)**: Train multiple networks with different initializations. Prediction variance across ensemble members estimates epistemic uncertainty.

**Monte-Carlo Dropout (MCDO)**: Apply dropout at test time and average over multiple forward passes. Variance approximates Bayesian posterior uncertainty.

**Bayesian Neural Networks (BNNs)**: Infer distributions over weights using variational inference. Provides principled uncertainty quantification but computationally expensive.

**Evidential Deep Learning (EDL)**: Directly outputs parameters of a Dirichlet distribution, enabling uncertainty estimation without ensembles or multiple passes.

**Surprise Adequacy (SA)**: Measures input novelty relative to training data:
- **DSA (Distance-based SA)**: Minimum distance to training activations
- **LSA (Likelihood-based SA)**: Likelihood under training activation distribution

**Application to DeepSample**: These UQ methods generate auxiliary variables (confidence, entropy, SA scores) that guide sampling toward uncertain or novel inputs likely to be mispredicted.

## 2.4 Related Work in DNN Testing

### 2.4.1 Coverage-Based Testing

**Neuron Coverage** (Pei et al., 2017): Measures percentage of neurons activated above threshold. Analogous to code coverage but lacks correlation with bug-finding effectiveness.

**DeepXplore** (Pei et al., 2017): Differential testing using multiple models. Generates inputs that produce divergent predictions, revealing potential errors.

**Limitations**: Coverage metrics don't account for operational input distribution; may generate unrealistic test cases.

### 2.4.2 Metamorphic Testing

**Metamorphic Relations**: Define expected output relationships under input transformations (e.g., rotation invariance for image classifiers).

**Advantages**: Enables testing without ground-truth labels by checking consistency properties.

**Limitations**: Requires domain-specific metamorphic relations; doesn't provide accuracy estimates.

### 2.4.3 Operational Profile Testing

**Statistical Testing**: Samples inputs according to operational usage distribution to estimate field reliability.

**DeepSample Positioning**: Extends operational profile testing with adaptive sampling strategies and formal statistical estimators, bridging classical software testing and modern ML evaluation.

## 2.5 Gap Analysis and Research Motivation

Existing approaches exhibit critical limitations:

1. **Cost-Effectiveness Gap**: Traditional methods require large labeled datasets; SubLIME/FLUID focus on benchmarks rather than operational testing
2. **Statistical Rigor Gap**: Coverage-based and metamorphic testing lack unbiased accuracy estimators with confidence intervals
3. **Flexibility Gap**: Current methods don't offer tunable trade-offs between accuracy estimation and failure detection
4. **Auxiliary Variable Gap**: Limited exploration of how different uncertainty proxies (confidence, entropy, SA) affect sampling efficiency

**DeepSample addresses these gaps** by providing a unified framework with multiple sampling strategies, formal statistical guarantees, and systematic auxiliary variable analysis—enabling practitioners to optimize testing for their specific objectives and constraints.
