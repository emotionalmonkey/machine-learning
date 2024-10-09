"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    ## This algorithm is numerically unstable when testing with netflix data
    #mu, var, p = mixture    
    #col = np.count_nonzero(X > 0, axis=1).reshape((-1,1))
    #X_n = (np.where(X > 0, X, np.nan)) # add nan where zero to exclude from computation
    #diff = (X_n[:,None] - mu) ** 2    
    #diff = np.where(np.isnan(diff) == False, diff, 0)
    #E = p * np.exp(-diff.sum(axis=-1) / (2 * var)) / (2 * np.pi * var) ** (0.5 * col)
    #post = E/E.sum(axis=1).reshape((-1,1))
    #LL = np.log(E.sum(axis=1)).sum()    
    #return post, LL

    mu, var, p = mixture    
    col = np.count_nonzero(X > 0, axis=1).reshape((-1,1))
    X_n = (np.where(X > 0, X, np.nan)) # add nan where zero to exclude from computation
    diff = (X_n[:,None] - mu) ** 2    
    diff = np.where(np.isnan(diff) == False, diff, 0)
    E = np.log(p + 1e-16) + (-diff.sum(axis=-1) / (2 * var))- (0.5 * col) * np.log(2 * np.pi * var)
    E_Log = logsumexp(E, axis=1).reshape(-1,1)
    post = np.exp(E - E_Log)
    LL = E_Log.sum()    
    return post, LL


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float=.25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """  
    p = post.sum(axis=0) / X.shape[0]   
    mu = mixture.mu
    mu = mu.astype(float) 
    
    indicator_matrix = np.where(X == 0, X, 1)  
    mu_hat = post.T @ X
    mu_indicator = post.T @ indicator_matrix
    nonzero_indices = np.where(mu_indicator > 0)
    mu[nonzero_indices] = mu_hat[nonzero_indices] / mu_indicator[nonzero_indices] # update only observed values

    X_n = (np.where(X > 0, X, np.nan)) # add nan where zero to exclude from computation
    diff = (X_n[:,None] - mu) ** 2    
    diff = np.where(np.isnan(diff) == False, diff, 0) # add zero where nan again
    var = diff.sum(axis=-1) * post    
    var = var.sum(axis=0) / mu_indicator.sum(axis=1)
    var = np.maximum(var, [min_variance] * var.shape[0])

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
        mixture = mstep(X, post, mixture)        
    
    return mixture, post, LL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mu, var, p = mixture  
    X_pred = X.copy()

    # Becuase estep function cannot be accessed on Grader
    col = np.count_nonzero(X > 0, axis=1).reshape((-1,1))
    X_n = (np.where(X > 0, X, np.nan)) # add nan where zero to exclude from computation
    diff = (X_n[:,None] - mu) ** 2    
    diff = np.where(np.isnan(diff) == False, diff, 0)
    E = np.log(p + 1e-16) + (-diff.sum(axis=-1) / (2 * var))- (0.5 * col) * np.log(2 * np.pi * var)
    E_Log = logsumexp(E, axis=1).reshape(-1,1)
    post = np.exp(E - E_Log)
    
    nonzero_indices = np.where(X == 0)   
    predictions = post @ mu
    X_pred[nonzero_indices] = predictions[nonzero_indices]
    
    return X_pred

## Testing
#import common
#import time
#import os
#X = np.loadtxt(os.path.join(os.path.dirname(__file__),"test_incomplete.txt"))
#K = 4
#seed = 0     
#mixture, post = common.init(X, K, seed)    
#mixture, post, cost = run(X, mixture, post)     
#print(mixture)
#X_pred = fill_matrix(X,mixture)
#X_gold = np.loadtxt(os.path.join(os.path.dirname(__file__),"test_complete.txt"))
#print("RMSE",common.rmse(X_gold, X_pred))
