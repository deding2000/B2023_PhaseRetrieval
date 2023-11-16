#%%
import numpy as np
from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp

n = 2
m = 4
A = np.fft.fft(np.eye(m))
A = A[:,np.random.choice(A.shape[0], n, replace=False)]
A = np.array([[ 1,1],
 [ 1,-1j],
 [ 1,-1],
 [ 1,1j]])
print(A)
repeats = 2000
success = np.zeros(repeats)
for i in range(repeats):
    x0 = np.random.normal(0,1,n) + 1j*np.random.normal(0,1,n)
    b = abs(A@x0)
    xhat = np.random.normal(0,1,n)+1j*np.random.normal(0,1,n)
    x = cp.Variable(n, complex=True)
    prob = cp.Problem(cp.Maximize(cp.real(inp(x,xhat))),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])
    prob.solve(verbose=False,solver="ECOS")
    xsol = x.value
    alpha = inp(x0,xsol)/(inp(xsol,xsol))
    xsol = alpha * xsol
    error = la.norm(x0-xsol)**2/la.norm(x0)**2
    if error <  1e-5:
        success[i] += 1

print(np.sum(success)/repeats)
# %%
