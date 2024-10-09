import numpy as np
import kmeans
import common
import naive_em
import em
import os

# TODO: Your code here
def run_K_means(): 
    print("K-means")
    for k in [1,2,3,4]:
        C = 0        
        for s in [0,1,2,3,4]:
            mixture, post = common.init(X, k, s)
            mixture, post, cost = kmeans.run(X, mixture, post)           
            C = min(C,cost) if C != 0 else cost
            #common.plot(X, mixture, post, name,s)        
        print("K=",k,", Min Cost=",C) 

def run_EM(name, fun, X, K=[1,2,3,4]):    
    seed = [0,1,2,3,4]
    print(name)    
    BIC = np.zeros(len(K))
    mixtures = [0] * len(K)
    for i in range(len(K)):
        log_LL = 0        
        for s in seed:
            mixture, post = common.init(X, K[i], s)
            mixture, post, LL = fun(X, mixture, post)            
            if log_LL == 0 or LL > log_LL:
                log_LL = LL
                mixtures[i] = mixture

            bic = common.bic(X,mixture,LL)
            if BIC[i] == 0 or bic > BIC[i]:
                BIC[i] = bic 
            
            #common.plot(X, mixture, post, name,s)        
        print("K=",K[i],", Max log-likelihood =",log_LL)        
    
    print("Best BIC=", max(BIC)) # Bayesian Information Criterion
    print("Best K=", K[np.argmax(BIC)]) # Best K is the one that produces optimal BIC (highest BIC)
    return mixtures[np.argmax(BIC)]


X = np.loadtxt(os.path.join(os.path.dirname(__file__),"toy_data.txt"))
run_K_means()
run_EM('Naive EM',naive_em.run, X)

"""
RESULT
-------
K-means
K= 1 , Min Cost= 5462.297452340002
K= 2 , Min Cost= 1684.9079502962372
K= 3 , Min Cost= 1329.59486715443
K= 4 , Min Cost= 1035.499826539466

Naive EM
K= 1 , Max log-likelihood = -1307.2234317600935
K= 2 , Max log-likelihood = -1175.7146293666797
K= 3 , Max log-likelihood = -1138.890899687267
K= 4 , Max log-likelihood = -1138.601175699485
Best BIC= -1169.2589347355092
Best K= 3
"""

"""
Reporting log likelihood values on Netflix data
Now, run the EM algorithm on the incomplete data matrix from Netflix ratings netflix_incomplete.txt. 
As before, please use seeds from [0,1,2,3,4] and report the best log likelihood you achieve with K = 1 and K = 12.

This may take on the order of a couple minutes for K = 12.
Report the maximum likelihood for each K using seeds [0,1,2,3,4]: 
______________________________________________________________________________
Test the accuracy of your predictions against actual target values by loading the complete matrix X_gold = np.loadtxt('netflix_complete.txt') 
and measuring the root mean squared error between the two matrices using common.rmse(X_gold, X_pred). 
Use your best mixture for from the first question of this tab to generate the results. 
"""

X = np.loadtxt(os.path.join(os.path.dirname(__file__),"netflix_incomplete.txt"))
mixture = run_EM('EM',em.run,X,[1,12])
X_pred = em.fill_matrix(X,mixture)
X_gold = np.loadtxt(os.path.join(os.path.dirname(__file__),"netflix_complete.txt"))
print("RMSE",common.rmse(X_gold, X_pred))

"""
RESULT
-------
EM
K= 1 , Max log-likelihood = -1521060.9539852478
K= 12 , Max log-likelihood = -1389858.8912802064
Best BIC= -1440988.9803814057
Best K= 12
RMSE 0.47587338036521215
"""
