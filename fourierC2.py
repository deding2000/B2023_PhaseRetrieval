#%%
import numpy as np
from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp

n = 2
m = 2
A = np.fft.fft(np.eye(m))
A = A[:,np.random.choice(A.shape[0], n, replace=False)]
repeats = 100
angles = np.pi/180*np.array([0,10,20,30,40,45,50,60,70,80,90])
success_sim = np.zeros(len(angles))
success_true = np.zeros(len(angles))
success_uni = np.zeros(len(angles))
for k, beta in enumerate(angles):
    success_true[k] = 1-2*beta/np.pi
    for i in range(repeats):
        x0 = np.random.normal(0,1,n) + 1j*np.random.normal(0,1,n)
        xhat = init_angle(x0, beta)
        b_f = abs(A@x0)
        # Solve fourier problem
        x = cp.Variable(n, complex=True)
        prob = cp.Problem(cp.Maximize(cp.real(inp(x,xhat))),[cp.norm(cp.multiply(A @ x,1/b_f), "inf") <= 1])
        prob.solve(verbose=False,solver="ECOS")
        xsol = x.value
        alpha = inp(x0,xsol)/(inp(xsol,xsol))
        xsol = alpha * xsol
        error = la.norm(x0-xsol)**2/la.norm(x0)**2
        if error <  1e-5:
            success_sim[k] += 1
        
plt.plot(angles*180/np.pi, success_sim*100/repeats,'r')
plt.plot(angles*180/np.pi, success_true*100,'b')

