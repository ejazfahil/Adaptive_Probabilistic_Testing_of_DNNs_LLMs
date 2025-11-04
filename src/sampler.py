"""Adaptive sampler. 2025-11-04"""
import numpy as np
from typing import Callable,Tuple

class AdaptiveSampler:
    def __init__(self,model_fn:Callable,shape:Tuple,step=0.01,seed=42):
        self.f=model_fn; self.shape=shape; self.step=step; self.rng=np.random.default_rng(seed)
    def _score(self,x):
        out=self.f(x); p=np.exp(out)/np.exp(out).sum()
        return float(-np.sum(p*np.log(p+1e-10)))
    def sample(self,n,x0=None):
        x=x0 if x0 is not None else self.rng.normal(size=self.shape)
        s=self._score(x); samples=[x.copy()]
        for _ in range(n-1):
            xn=np.clip(x+self.rng.normal(scale=self.step,size=self.shape),0,1)
            sn=self._score(xn)
            if sn>s or self.rng.random()<np.exp(sn-s): x,s=xn,sn
            samples.append(x.copy())
        return samples
