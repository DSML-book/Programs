""" CMCtestloss.py """
import numpy as np
from numpy.random import rand, randn
from numpy.linalg import solve
import matplotlib.pyplot as plt

def generate_data(beta, sig, n):
    u = rand(n, 1)
    y = (u ** np.arange(0, 4)) @ beta + sig * randn(n, 1)
    return u, y

beta = np.array([[10, -140, 400, -250]]).T
n = 100
sig = 5
betahat = {}
plt.figure(figsize=[6,3])

for N in range(0,100):
    max_p = 8
    p_range = np.arange(1, max_p + 1, 1)
    u, y = generate_data(beta, sig, n)
    
    MSE = []
    X = np.ones((n, 1))
    
    for p in p_range:
        if p > 1:
            X = np.hstack((X, u**(p-1)))
            
        betahat[p] = solve(X.T @ X, X.T @ y)
        y_hat = X @ betahat[p]  # predictions
        MSE.append(np.sum((y - y_hat)**2/n))
    
    plt.plot(p_range, MSE,'C0',alpha=0.1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.xticks(ticks=p_range)
plt.xlabel('Number of parameters $p$')
plt.ylabel('Test loss')
plt.tight_layout()
plt.savefig('MSErepeatpy.pdf',format='pdf')
plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%