#%%
import numpy as np
from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp

n = 4
m = 8
A = np.fft.fft(np.eye(m))
A = A[:,np.random.choice(A.shape[0], n, replace=False)]
print(A)
x0 = np.random.normal(0,1,n)
b = abs(A@x0)
x = np.zeros(2)
repeats = 1000
success = np.zeros(repeats)
for i in range(repeats):
    xhat = np.random.normal(0,1,n)
    x = cp.Variable(n, complex=False)
    prob = cp.Problem((cp.Maximize(x.T@xhat)),[cp.norm((A @ x*1/b), "inf") <= 1])
    prob.solve(verbose=False,solver="ECOS")
    xsol = x.value
    alpha = (x0.T@xsol)/(xsol.T@xsol.T)
    xsol = alpha * xsol
    error = la.norm(x0-xsol)**2/la.norm(x0)**2
    if error <  1e-5:
        success[i] += 1

print(np.sum(success)/repeats)

#%%
print(angle(A[0,:],A[1,:])*180/np.pi)
print(angle(A[0,:],A[2,:])*180/np.pi)
print(angle(A[1,:],A[2,:])*180/np.pi)


# %%
