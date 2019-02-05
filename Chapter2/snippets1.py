"""
Rob Salomone Mon Oct 22 15:45:13 2018
"""
# %%
import numpy as np
from numpy.random import rand, randn
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt


def generate_data(beta, sig, n):
    u = np.random.rand(n, 1)
    y = (u ** np.arange(0, 4)) @ beta + sig * np.random.randn(n, 1)
    return u, y


plt.figure(1), plt.clf()
np.random.seed(12)
beta = np.array([[10, -140, 400, -250]]).T

n = 100
sig = 5
u, y = generate_data(beta, sig, n)

xx = np.arange(np.min(u), np.max(u)+5e-3, 5e-3)
yy = np.polyval(np.flip(beta), xx)
plt.plot(xx, yy, 'k--')
plt.plot(u, y, 'k.', markersize=4)
# %%
max_p = 18
p_range = np.arange(1, max_p + 1, 1)
X = np.ones((n, 1))
betahat, trainloss = {}, {}

for p in p_range:  # k is polynomial degree
    if p > 1:
        X = np.hstack((X, u**(p-1)))  # add column to matrix

    betahat[p] = solve(X.T @ X, X.T @ y)
    trainloss[p] = (norm(y - X @ betahat[p])**2/n)

p = [2, 4, 16]  # number of parameters (p = k + 1)
for i in p:
    yy = np.polyval(np.flip(betahat[i]), xx)
    plt.plot(xx, yy)

# plt.xlabel(r'$u$')
# plt.ylabel(r'$h^{\mathcal{G}_p}(u)$')
# %%

plt.figure(2), plt.clf()

# generate test data
u_test, y_test = generate_data(beta, sig, n)

MSE = []
X_test = np.ones((n, 1))

for p in p_range:
    if p > 1:
        X_test = np.hstack((X_test, u_test**(p-1)))

    y_hat = X_test @ betahat[p]  # predictions
    MSE.append(np.sum((y_test - y_hat)**2/n))

plt.plot(p_range, MSE, 'b', p_range, MSE, 'bo')
plt.xticks(ticks=p_range)
plt.xlabel('Number of parameters $p$')
plt.ylabel('Test loss')

# %%
plt.figure(3), plt.clf()

beta = np.array([[10, -140, 400, -250]]).T
n = 100
sig = 5
u, y = generate_data(beta, sig, n)

K_vals = [5, 10, 100]  # number of blocks
cv = np.zeros((len(K_vals), max_p))
X = np.ones((n, 1))

for p in p_range:
    if p > 1:
        X = np.hstack((X, u**(p-1)))
    j = 0
    for K in K_vals:
        loss = []
        for k in range(1, K+1):
            # integer indices of test samples
            test_ind = ((n/K)*(k-1) + np.arange(1, n/K + 1) - 1).astype('int')
            train_ind = np.setdiff1d(np.arange(n), test_ind)

            X_train, y_train = X[train_ind, :], y[train_ind, :]
            X_test, y_test = X[test_ind, :], y[test_ind]

            # fit model and evaluate test loss
            betahat = solve(X_train.T @ X_train, X_train.T @ y_train)
            loss.append(norm(y_test - X_test @ betahat) ** 2)

        cv[j, p-1] = sum(loss) / n
        j += 1

# %% for plotting (don't include in book)
msize = 10

plt.plot(p_range, cv[0, :], 'k-.', p_range, cv[0, :], 'k.', markersize=msize)
plt.plot(p_range, cv[1, :], 'r', p_range, cv[1, :], 'r.', markersize=msize)
plt.plot(p_range, cv[2, :], 'b--', p_range, cv[2, :], 'b.', markersize=msize)
plt.xticks(range(2, 19, 2))
