"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    mu, var, p = mixture  
    d = X.shape[1] 
   
    diff = ((X[:,None] - mu)**2).sum(axis=-1)
    E = p * np.exp(-diff / (2 * var)) / (2 * np.pi * var) ** (0.5 * d)
    post = E/E.sum(axis=1).reshape((-1,1))    
    LL = np.log(E.sum(axis=1)).sum()    
    return post, LL

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    sum_p = post.sum(axis=0)
    p = sum_p / n
    
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / sum_p[j]
        diff = ((mu[j] - X)**2).sum(axis=1) @ post[:, j]        
        var[j] = diff / (d * sum_p[j])

    return GaussianMixture(mu, var, p)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_LL = None   
    LL = None   
    while prev_LL is None or LL - prev_LL > 1e-06 * abs(LL):
        prev_LL = LL
        post, LL = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, LL


## Testing
#import common
#import os
#X = np.loadtxt(os.path.join(os.path.dirname(__file__),"toy_data.txt"))
#K = 3
#seed = 0   
#mixture, post = common.init(X, K, seed)    
#mixture, post, cost = run(X, mixture, post)        
#common.plot(X, mixture, post, 'Naive EM')
#print("K=",K,", Mixture=",mixture)

