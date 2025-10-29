"""Coverage metrics. 2025-10-29"""
import numpy as np
from typing import List

def neuron_coverage(acts: List[np.ndarray], threshold=0.5):
    if not acts: return 0.0
    all_a = np.stack(acts)
    return (all_a.max(axis=0)>threshold).sum()/all_a.shape[1]

def top_k_coverage(acts: List[np.ndarray], k=5):
    covered=set()
    for a in acts: covered.update(np.argsort(a)[-k:].tolist())
    return len(covered)/acts[0].shape[0] if acts else 0.0
