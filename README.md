# Adaptive Probabilistic Testing of DNNs & LLMs

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

A research prototype exploring **adaptive, probabilistic test-input generation**
for deep neural networks and LLMs — using uncertainty-guided sampling to drive a
model into under-tested regions of its input space, measured by neuron-coverage
metrics.

> **Status:** early research prototype. The core sampler and coverage metrics are
> implemented as standalone, importable functions; the end-to-end testing loop
> (real model, paraphrase-consistency oracle, coverage-vs-iterations study) is
> the planned next stage. This README documents the approach and roadmap honestly
> rather than reporting results that do not yet exist.

---

## Overview & Aim

Standard test sets exercise only a narrow slice of a model's behavior. The
research question here is: **can we generate inputs that are deliberately
informative** — inputs that push the model toward high-uncertainty, low-coverage
regions where bugs and inconsistencies hide — rather than sampling uniformly?

The prototype combines two ingredients drawn from the probabilistic-testing
literature (DeepSample-style adaptive generation, neuron-coverage criteria):

1. an **uncertainty-guided adaptive sampler** that performs a Metropolis-style
   walk through input space, biased toward high model uncertainty; and
2. **coverage metrics** that quantify how much of the network's activation
   structure a set of inputs has exercised.

---

## Methodology / How It Works

### Adaptive sampler (entropy-guided MCMC)

[`AdaptiveSampler`](src/sampler.py) wraps any model function
`model_fn(x) → logits` and runs an MCMC-style random walk whose objective is the
**predictive entropy** of the model's softmax output:

$$\mathcal{H}(x) = -\sum_{c} p_c(x)\,\log p_c(x),\qquad p(x) = \mathrm{softmax}\bigl(\text{model\_fn}(x)\bigr).$$

High entropy means the model is *uncertain* — exactly the region we want to test.
From a current input $x$ a perturbed candidate $x' = \mathrm{clip}(x + \mathcal{N}(0,\sigma),\,0,1)$
is proposed and accepted with a Metropolis-style rule:

$$x \leftarrow x' \quad\text{if}\quad \mathcal{H}(x') > \mathcal{H}(x)\ \ \text{or}\ \ u < e^{\mathcal{H}(x') - \mathcal{H}(x)},\ \ u\sim U(0,1),$$

so the chain climbs toward high-uncertainty inputs while still escaping local
maxima. The walk returns the full trajectory of accepted samples; the step size
`σ` (`step`) and RNG `seed` are configurable for reproducibility.

### Coverage metrics

[`coverage.py`](src/coverage.py) provides two activation-coverage measures over a
list of per-input activation vectors:

- **Neuron coverage** — fraction of neurons whose activation exceeds a threshold
  on *at least one* input:
  $$\mathrm{NC} = \frac{|\{j : \max_i a_{ij} > \tau\}|}{N_{\text{neurons}}}$$
- **Top-$k$ coverage** — fraction of neurons that ever appear among the top-$k$
  most-activated on some input (a stricter, ranking-based criterion).

Together these let an experiment ask: *does the adaptive sampler cover more of the
network, faster, than uniform sampling?*

### Consistency oracle (planned)

The research note frames an **LLM consistency** oracle —
$P\bigl(f(x) = f(\text{paraphrase}(x))\bigr) \approx 1$ — as a metamorphic test:
adaptively search for paraphrase pairs on which the model disagrees. This oracle
is described in [`docs/research.md`](docs/research.md) and is the next component
to implement.

---

## Tech Stack & Tools

- **Python 3.11+**
- **NumPy** — softmax/entropy scoring, activation stacking, coverage computation
- Typing-annotated, dependency-light building blocks (model-framework agnostic:
  any `model_fn(x) → logits` works)

---

## Project Structure

```
Adaptive_Probabilistic_Testing_of_DNNs_LLMs/
├── src/
│   ├── sampler.py      # AdaptiveSampler: entropy-guided Metropolis walk
│   └── coverage.py     # neuron_coverage() and top_k_coverage()
└── docs/
    └── research.md     # concepts: neuron coverage, DeepSample, LLM consistency
```

---

## Key Features

- **Model-agnostic** — the sampler takes any callable returning logits, so it
  applies to DNN classifiers and (with a tokenized interface) LLMs.
- **Uncertainty-seeking** — predictive entropy as the acceptance objective drives
  generation toward informative, bug-revealing inputs.
- **Two coverage criteria** — threshold-based neuron coverage and ranking-based
  top-$k$ coverage for measuring test adequacy.
- **Reproducible** — seeded RNG and explicit step size.

---

## Results

No empirical results are reported yet — this is a research prototype. The intended
study is a **coverage-vs-iterations** comparison: plotting neuron / top-$k$
coverage achieved by the adaptive sampler against a uniform-sampling baseline on a
fixed model, plus a consistency-violation rate from the planned paraphrase oracle.
Numbers will be added only once that experiment has genuinely been run.

---

## Getting Started

```python
import numpy as np
from src.sampler import AdaptiveSampler
from src.coverage import neuron_coverage, top_k_coverage

# model_fn: np.ndarray -> logits (np.ndarray)
sampler = AdaptiveSampler(model_fn, shape=(28, 28), step=0.02, seed=42)
trajectory = sampler.sample(n=500)

# collect activations from your model for each sampled input, then:
print("neuron coverage:", neuron_coverage(activations, threshold=0.5))
print("top-k coverage:", top_k_coverage(activations, k=5))
```

---

## Challenges

- **Designing the right objective** — predictive entropy is a proxy for
  informativeness; calibrating step size so the chain explores without
  random-walking is non-trivial.
- **Coverage as a moving target** — neuron coverage saturates; top-$k$ coverage
  resists saturation but is sensitive to `k`.
- **A trustworthy oracle for LLMs** — paraphrase-consistency needs a paraphraser
  that preserves meaning, itself a modeling problem.

## Future Work

- Implement the paraphrase-consistency oracle for LLM testing.
- Add a uniform-sampling baseline and the coverage-vs-iterations experiment.
- Plug in a concrete DNN (e.g. an MNIST/CIFAR classifier) end to end.
- Surprise-adequacy / activation-based coverage beyond threshold + top-$k$.

## Conclusion

This prototype assembles the two primitives of adaptive probabilistic model
testing — an uncertainty-guided generator and activation-coverage metrics — into
clean, reusable code, and lays out a concrete research plan for turning them into
a measured testing methodology for DNNs and LLMs.
