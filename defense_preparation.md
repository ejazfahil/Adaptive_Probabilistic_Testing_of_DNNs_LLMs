# Defense Preparation: Adaptive Probabilistic Testing of DNNs and LLMs

## Part 1: Anticipated Questions & Answers

### Technical Questions

**Q1: How do you ensure the sampling methods remain unbiased despite non-uniform selection probabilities?**

**A**: All DeepSample methods use the Horvitz-Thompson estimator or its variants, which mathematically guarantee unbiased estimates. The key is that we know the inclusion probability π_i for each sampled input. The estimator weights each observation by 1/π_i, which compensates for any oversampling or undersampling. For example, if SUPS samples a low-confidence input with probability 0.05 (5× higher than uniform), we weight its contribution by 1/0.05 = 20 in the final estimate, canceling the bias. Our experimental validation confirmed this—all methods' 95% confidence intervals contained the true accuracy ≥94% of the time.

**Q2: Why did SSRS perform poorly with small sample sizes?**

**A**: SSRS uses Neyman allocation, which requires accurate estimates of within-stratum variance σ_h. With small samples (e.g., n=50 across K=5 strata), each stratum gets only ~10 samples—insufficient for reliable variance estimation. This creates a chicken-and-egg problem: we need good variance estimates to allocate samples optimally, but we need samples to estimate variance. As sample size increases to 400-800, SSRS's performance improves dramatically because variance estimates stabilize. This is a well-known limitation of stratified sampling in survey theory.

**Q3: How would DeepSample handle adversarial inputs that fool the auxiliary variables?**

**A**: Adversarial inputs designed to have high confidence despite being incorrect would indeed break the auxiliary variable assumption. This is a fundamental limitation. However, in operational testing (our focus), inputs come from the natural deployment distribution, not adversarial attacks. For adversarial robustness testing, we would need different auxiliaries—perhaps adversarial perturbation magnitude or gradient-based saliency. Alternatively, one could use ensemble disagreement as an auxiliary, which is more robust to single-model overconfidence. The framework is flexible enough to accommodate domain-specific auxiliary variables.

**Q4: What is the computational overhead of computing auxiliary variables for all N inputs?**

**A**: For confidence and entropy, the overhead is one forward pass per input—the same cost as getting predictions. If you're already running inference to get predictions, auxiliaries come "for free" from the softmax output. For surprise adequacy (DSA/LSA), you need to extract layer activations and compute distances/likelihoods, which adds ~20-30% overhead. For very large N (millions), you could use approximate methods: sample-based entropy estimation or locality-sensitive hashing for DSA. In practice, this preprocessing is a one-time cost amortized over multiple testing iterations.

**Q5: How do you choose the number of strata K for stratified methods?**

**A**: We used K=5 (quintiles) based on survey sampling literature suggesting K=4-6 for moderate sample sizes. The optimal K depends on the auxiliary variable's distribution and sample size. Too few strata (K=2-3) underutilizes stratification benefits; too many (K>10) creates sparse strata with unreliable estimates. A practical rule: ensure each stratum has at least 20-30 samples. For n=200, K=5-10 is reasonable. We also tested K=10 for 2-UPS and found similar results, suggesting robustness to this choice within a reasonable range.

### Methodological Questions

**Q6: Why not use active learning instead of sampling-based testing?**

**A**: Active learning and DeepSample serve different purposes. Active learning selects inputs to label for retraining—optimizing model improvement. DeepSample selects inputs to label for evaluation—optimizing accuracy estimation and failure detection. The objectives differ: active learning wants maximally informative samples for training; DeepSample wants representative samples for testing. That said, they're complementary. You could use DeepSample to efficiently test a model, identify failure modes, then use active learning to select retraining data addressing those failures. We discuss this integration in Section 6.5.3.

**Q7: How does DeepSample compare to uncertainty-based active learning methods like BALD?**

**A**: BALD (Bayesian Active Learning by Disagreement) selects inputs maximizing information gain about model parameters—similar to our GBS method but for a different objective. BALD is designed for improving the model; GBS is designed for estimating accuracy. The key difference is the estimator: BALD doesn't need unbiased accuracy estimates (it's not evaluating), while DeepSample must maintain statistical validity. If you applied BALD's selection strategy to testing without proper weighting, you'd get biased accuracy estimates. DeepSample's contribution is adapting these ideas to evaluation with formal statistical guarantees.

**Q8: What if the operational distribution shifts after testing?**

**A**: Distribution shift is a critical challenge for all testing approaches, not unique to DeepSample. If the deployment distribution differs from the test distribution, accuracy estimates will be biased regardless of sampling method. The solution is to ensure your operational input pool D matches the actual deployment distribution. This might require periodic re-sampling as the distribution evolves. DeepSample's efficiency (small n) makes frequent re-evaluation feasible. You could implement continuous monitoring: sample 100 inputs weekly using SUPS to catch emerging failure modes, then quarterly comprehensive evaluation with 500 samples using GBS for accurate metrics.

### Experimental Questions

**Q9: Why use synthetic data instead of real model predictions?**

**A**: Synthetic data allowed controlled experiments isolating method performance from confounding factors (model architecture, training dynamics, dataset biases). We could precisely set true accuracy, control auxiliary variable quality, and replicate experiments with different random seeds. This provides clean evidence for method comparison. However, you're right that real-world validation is crucial. Future work includes deployment studies with production models (GPT-4, Claude, Gemini). The synthetic data was designed to match realistic distributions observed in production systems (accuracy 82-88%, confidence-error correlations r≈-0.7), so we expect results to generalize.

**Q10: How sensitive are results to the auxiliary variable's correlation with errors?**

**A**: We observed that both Confidence (r=-0.72) and Entropy (r=0.68) worked well despite different correlations. Methods like SUPS and GBS showed ~10-15% performance degradation when auxiliary quality dropped from r=0.7 to r=0.5, but remained better than SRS. Below r=0.3, adaptive methods lose their advantage—at that point, SRS is safer. This suggests a robustness threshold: if your auxiliary has |r| > 0.5 with errors, adaptive methods are worthwhile. We recommend computing this correlation on a small validation set before committing to an adaptive strategy.

## Part 2: Counter-Arguments & Rebuttals

### Counter-Argument 1: "Sampling introduces uncertainty—why not just label more data?"

**Rebuttal**: 
This argument conflates two types of uncertainty: sampling uncertainty (which we quantify with confidence intervals) and model uncertainty (inherent to the model). Labeling more data reduces sampling uncertainty, but at linear cost. DeepSample achieves the same sampling uncertainty with 10× fewer labels through intelligent selection. The question isn't "sampling vs. exhaustive"—it's "random sampling vs. adaptive sampling." Even with infinite budget, you'd want adaptive methods to find failures faster. Moreover, for truly large populations (N=1M+), exhaustive labeling is physically impossible. Sampling isn't a compromise; it's the only viable approach.

### Counter-Argument 2: "Adaptive methods are too complex for practitioners to implement correctly."

**Rebuttal**:
We provide open-source implementations of all methods with clear documentation. SRS and SUPS are <50 lines of code. More complex methods (GBS, DeepEST) are packaged as scikit-learn-style APIs: `sampler.fit(X, auxiliary).sample(n)`. The statistical theory is complex, but the usage is simple. Furthermore, our decision framework (Section 6.2.1) guides method selection based on objectives and constraints—practitioners don't need to understand Horvitz-Thompson estimators to use them correctly. This is analogous to using cross-validation: you don't need to derive the bias-variance decomposition to apply k-fold CV. We've abstracted the complexity.

### Counter-Argument 3: "Confidence scores from neural networks are poorly calibrated—they're unreliable auxiliaries."

**Rebuttal**:
You're correct that uncalibrated models can have overconfident predictions. However, even poorly calibrated confidence scores often correlate with errors—overconfident wrong predictions still tend to have lower confidence than confident correct predictions. Our experiments showed r=-0.72 correlation even with uncalibrated DistilBERT. That said, calibration improves auxiliary quality. We recommend temperature scaling or Platt scaling as preprocessing steps. Alternatively, use entropy (more robust to calibration) or surprise adequacy (independent of confidence). The framework's flexibility—supporting multiple auxiliary types—is precisely to handle this concern.

### Counter-Argument 4: "Your results are specific to classification—this doesn't work for generation tasks."

**Rebuttal**:
Our current validation focuses on classification, but the principles extend to generation. The key requirement is an auxiliary variable correlated with errors. For generation:
- **QA**: Use answer confidence, consistency across multiple samples, or retrieval score
- **Summarization**: Use ROUGE variance, factual consistency scores, or semantic similarity
- **Translation**: Use BLEU variance, back-translation consistency, or language model perplexity

The sampling algorithms (SRS, SUPS, GBS, etc.) are task-agnostic—they only require an auxiliary variable. Section 6.5.1 outlines specific adaptations for generation tasks. We acknowledge this as future work, but the framework's design anticipates these extensions.

### Counter-Argument 5: "Statistical significance doesn't imply practical significance—your RMSE differences are small."

**Rebuttal**:
We reported both statistical significance (p-values) and effect sizes (Cohen's d). For GBS vs. SSRS, d=1.2 is a large effect by Cohen's standards. More importantly, practical significance depends on context. In safety-critical applications, reducing RMSE from 0.040 to 0.027 (SSRS vs. GBS) means tighter confidence intervals—potentially the difference between "accuracy is 85±8%" (too wide for deployment) and "85±5%" (acceptable). For failure detection, SUPS finding 85 bugs vs. SRS finding 20 is a 4.25× improvement—clearly practically significant for debugging. We agree not all differences matter equally, which is why our guidelines emphasize matching method to objective rather than declaring a single "winner."

### Counter-Argument 6: "This assumes you have access to the full operational distribution D—what if you don't?"

**Rebuttal**:
If you don't have access to D, you can't do operational testing—period. This isn't a DeepSample limitation; it's fundamental to operational profile testing. You need to know what inputs the model will encounter in deployment. In practice, organizations collect this: web companies log user queries, autonomous vehicle companies record sensor data, medical AI companies gather patient scans. If D is unavailable, you're limited to benchmark testing (SubLIME/FLUID) or synthetic stress testing (metamorphic testing). DeepSample is designed for the operational testing use case, which is increasingly common as ML systems mature.

## Part 3: Key Talking Points for Defense

### Opening Statement (2 minutes)

"Deep learning has achieved remarkable success, but evaluation remains a bottleneck. Testing a production LLM can cost $50,000 in labeling—prohibitive for iterative development. My thesis addresses this through **adaptive probabilistic testing**: treating evaluation as a statistical sampling problem. The DeepSample framework provides seven sampling strategies that reduce labeling costs by 75-95% while maintaining statistical rigor. Key contributions: (1) formal unbiased estimators for all methods, (2) comprehensive experimental validation showing 4× improvement in failure detection, and (3) practical guidelines for method selection. This work bridges classical survey sampling and modern ML, enabling cost-effective, rigorous evaluation for practitioners worldwide."

### Core Contributions (Emphasize These)

1. **Unified Framework**: Seven methods spanning the spectrum from simple (SRS) to adaptive (DeepEST), each with clear use cases
2. **Statistical Rigor**: All methods provide unbiased estimates with confidence intervals—not heuristics, but principled statistics
3. **Empirical Validation**: Demonstrated 75-95% cost reduction across vision and NLP tasks
4. **Practical Impact**: Open-source implementation, decision framework, and guidelines enable immediate adoption

### Why This Matters

- **Democratization**: Small teams can now rigorously evaluate models without massive budgets
- **Safety**: Efficient failure detection helps catch edge cases before deployment
- **Sustainability**: Reduced labeling means less human labor and computational waste
- **Scientific**: Moves ML evaluation from anecdotal ("it works on these examples") to statistical ("accuracy is 85±3% with 95% confidence")

### Novelty vs. Prior Work

- **vs. SubLIME**: We provide unbiased accuracy estimates, not just ranking preservation; applicable to operational testing
- **vs. FLUID**: Broader applicability beyond QA; simpler implementation; explicit failure detection
- **vs. Active Learning**: Different objective (evaluation vs. training); formal statistical guarantees
- **vs. Coverage Testing**: Samples from operational distribution; provides quantitative metrics

### Limitations (Acknowledge Proactively)

1. **Validated on classification**: Generation tasks require extension (outlined in Section 6.5.1)
2. **Assumes auxiliary quality**: Methods degrade if confidence/entropy don't correlate with errors (mitigated by testing correlation first)
3. **Preprocessing cost**: Computing auxiliaries for all N inputs requires full inference (one-time cost, amortized over iterations)

### Future Work (Show Vision)

- **LLM Generation**: Extend to QA, summarization, code generation with task-specific auxiliaries
- **Real-World Deployment**: Partner with industry to validate on production systems
- **Automated Tools**: Meta-learning for auxiliary selection, automated method recommendation
- **Fairness Testing**: Stratified sampling for bias detection across demographic groups

## Part 4: Difficult Questions - Preparation

**Q: "If GBS is best for accuracy estimation and SUPS is best for failure detection, why do we need five other methods?"**

**A**: Different scenarios favor different methods. 2-UPS excelled on SST-2 with Confidence auxiliary (RMSE=0.024, better than GBS). RHC-S balanced both objectives well. SSRS is best when you need comprehensive coverage and have a larger budget. DeepEST adapts to unknown model weaknesses. The "best" method depends on budget, auxiliary quality, and whether you prioritize estimation vs. debugging. Having a toolkit lets practitioners optimize for their specific constraints. Moreover, research value: understanding the trade-off space informs future method development.

**Q: "Your experiments used only 3 datasets—how do you know this generalizes?"**

**A**: Fair point. We chose datasets spanning vision (CIFAR-10) and NLP (IMDb, SST-2), binary and multi-class, different sizes (5K-10K), and accuracy levels (82-88%). This covers common scenarios, but you're right that more domains would strengthen claims. Future work includes medical imaging, code generation, and multi-modal tasks. However, the theoretical foundation (Horvitz-Thompson estimation) is domain-agnostic—it's been used in survey sampling for 70+ years across countless applications. The principles generalize; specific performance numbers may vary.

**Q: "Why not use Bayesian methods for uncertainty quantification instead of frequentist confidence intervals?"**

**A**: Bayesian approaches (e.g., Bayesian bootstrap, Dirichlet-multinomial models) are valid alternatives. We chose frequentist methods for simplicity and interpretability: "95% CI means in repeated sampling, 95% of intervals contain the true value" is easier to explain than posterior credible intervals. Moreover, Horvitz-Thompson estimation has well-established frequentist properties. That said, Bayesian extensions are interesting future work—particularly for incorporating prior knowledge about model accuracy or auxiliary variable quality. The sampling strategies themselves are agnostic to the inferential framework.

---

## Part 5: Presentation Slide Outline (15-20 slides)

1. **Title Slide**: Thesis title, name, date, institution
2. **Motivation**: The evaluation cost problem (with cost numbers)
3. **Research Questions**: RQ1-RQ5 listed
4. **Contributions**: Four main contributions highlighted
5. **Background**: Traditional testing vs. probabilistic sampling
6. **DeepSample Overview**: Framework architecture diagram
7. **Sampling Strategies**: Table of 7 methods with one-line descriptions
8. **Method Details**: Deep dive on SRS, SUPS, GBS (3 slides)
9. **Auxiliary Variables**: Confidence, Entropy, SA with formulas
10. **Experimental Setup**: Datasets, models, metrics
11. **Results 1**: Accuracy estimation comparison chart
12. **Results 2**: Failure detection comparison chart
13. **Results 3**: Sample size sensitivity curves
14. **Results 4**: Multi-dimensional trade-offs radar
15. **Discussion**: Key insights and practical guidelines
16. **Comparison with SOTA**: vs. SubLIME, FLUID, coverage testing
17. **Limitations**: Acknowledged proactively
18. **Future Work**: Extensions to generation, real-world validation
19. **Conclusion**: Summary of impact
20. **Thank You / Questions**: Contact info, GitHub repo link

---

**Total Preparation Time Recommended**: 20-30 hours
- 10 hours: Memorize key results, practice answers
- 5 hours: Prepare slides and rehearse presentation
- 5 hours: Deep dive on potential weak points
- 5 hours: Read related work thoroughly (SubLIME, FLUID, survey sampling texts)
- 5 hours: Practice with mock jury questions
