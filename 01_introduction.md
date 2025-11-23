# Chapter 1: Introduction

## 1.1 Motivation and Problem Statement

Deep Neural Networks (DNNs) and Large Language Models (LLMs) have achieved remarkable success across diverse domains, from computer vision to natural language processing. However, their deployment in safety-critical applications—such as autonomous vehicles, medical diagnosis, and financial systems—demands rigorous evaluation to ensure reliability and trustworthiness. Traditional testing approaches face a fundamental challenge: **the cost of comprehensive evaluation scales linearly with dataset size**, making exhaustive testing prohibitively expensive.

The conventional paradigm requires manually labeling thousands or millions of test samples to obtain statistically reliable accuracy estimates. For instance, evaluating a production-ready LLM on a benchmark like MMLU (Massive Multitask Language Understanding) involves testing on 15,000+ questions across 57 subjects. With human annotation costs ranging from $0.10 to $5.00 per sample depending on task complexity, comprehensive evaluation can cost tens of thousands of dollars per model iteration.

Moreover, the rapid pace of model development exacerbates this problem. Organizations frequently retrain models, fine-tune on new data, or deploy updated architectures. Each iteration requires re-evaluation, creating an unsustainable testing bottleneck. The research community needs **cost-effective testing methodologies** that maintain statistical rigor while dramatically reducing labeling requirements.

## 1.2 The DeepSample Framework: A Paradigm Shift

This thesis presents **DeepSample**, a comprehensive framework for adaptive probabilistic testing of DNNs and LLMs. DeepSample reframes model evaluation as a **statistical sampling problem** rather than an exhaustive enumeration task. By intelligently selecting a small, representative subset of test inputs and applying principled statistical estimators, DeepSample achieves three critical objectives:

1. **Minimal Test Set Size**: Reduce labeling costs by orders of magnitude through intelligent sampling
2. **Unbiased Accuracy Estimates**: Provide statistically valid performance metrics with quantified confidence intervals
3. **Effective Failure Detection**: Expose model mispredictions efficiently to guide debugging and improvement

The framework encompasses seven distinct sampling strategies—Simple Random Sampling (SRS), Simple Unequal Probability Sampling (SUPS), RHC-Sampling (RHC-S), Stratified Simple Random Sampling (SSRS), Gradient-Based Sampling (GBS), Two-stage Unequal Probability Sampling (2-UPS), and DeepEST—each optimized for different testing objectives and operational constraints.

## 1.3 Research Contributions

This thesis makes the following contributions to the field of DNN/LLM testing:

### 1.3.1 Methodological Contributions

- **Comprehensive Sampling Framework**: A unified family of seven probabilistic sampling techniques for DNN/LLM evaluation, each with distinct trade-offs between accuracy estimation and failure detection
- **Auxiliary Variable Analysis**: Systematic investigation of uncertainty proxies (confidence scores, prediction entropy, surprise adequacy metrics) for guiding sample selection
- **Statistical Rigor**: Formal unbiased estimators with confidence intervals for all sampling methods, ensuring valid inference despite non-uniform selection probabilities

### 1.3.2 Empirical Contributions

- **Extensive Experimental Validation**: Evaluation across vision (CIFAR-10) and NLP tasks (IMDb sentiment, SST-2) using DistilBERT and convolutional architectures
- **Sample Size Sensitivity Analysis**: Characterization of performance across budgets ranging from 50 to 800 labeled samples
- **Comparative Benchmarking**: Head-to-head comparison of all methods on accuracy estimation error (RMSE) and misprediction detection rates

### 1.3.3 Practical Contributions

- **Practitioner Guidelines**: Decision framework for selecting appropriate sampling strategies based on testing objectives, budget constraints, and domain characteristics
- **Open Implementation**: Reproducible Python codebase with synthetic datasets for research community adoption
- **Cost-Benefit Analysis**: Quantification of labeling cost savings (up to 95% reduction) while maintaining statistical validity

## 1.4 Integration with State-of-the-Art Research (2024)

DeepSample aligns with and extends recent advances in efficient model evaluation:

**SubLIME Framework** (2024): While SubLIME focuses on benchmark subset selection for LLM leaderboards using clustering and quality-based sampling, DeepSample provides a broader statistical foundation with formal unbiased estimators applicable to operational testing scenarios.

**FLUID BENCHMARKING** (2024): Inspired by psychometric Item Response Theory (IRT), FLUID adapts test difficulty to model capability. DeepSample complements this by offering multiple sampling strategies beyond adaptive item selection, including stratified and gradient-based approaches.

**Uncertainty Quantification Methods** (2024): Recent advances in Deep Ensembles, Monte-Carlo Dropout, and Evidential Deep Learning provide robust uncertainty estimates. DeepSample leverages these as auxiliary variables to guide sampling, creating a synergy between UQ research and efficient testing.

## 1.5 Thesis Organization

The remainder of this thesis is structured as follows:

**Chapter 2: Background and Related Work** surveys traditional DNN/LLM evaluation approaches, probabilistic sampling theory foundations, and state-of-the-art methods from 2024 including SubLIME, FLUID, and advanced uncertainty quantification techniques.

**Chapter 3: The DeepSample Framework** presents the core methodology, detailing all seven sampling strategies, auxiliary variable computation, and statistical estimators for unbiased accuracy assessment.

**Chapter 4: Experimental Design** describes the synthetic datasets, model architectures, evaluation metrics, and experimental protocols used to validate the framework.

**Chapter 5: Results and Analysis** reports comprehensive experimental findings, including accuracy estimation performance, misprediction detection rates, and sample size sensitivity across all methods.

**Chapter 6: Discussion** interprets results, provides practical guidelines for method selection, discusses limitations, and outlines future research directions including extensions to question answering and text summarization tasks.

**Chapter 7: Conclusion** summarizes contributions, reflects on the impact of probabilistic operational testing, and presents final remarks on the future of cost-effective DNN/LLM evaluation.
