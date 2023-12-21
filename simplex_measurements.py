#%%
import numpy as np
from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp
# Real and simple case with different angles
n = 3
m = 4
A1 = np.array([[1,0],[-np.cos(2*np.pi/3),-np.sin(2*np.pi/3)],[-np.cos(2*np.pi/3),np.sin(2*np.pi/3)]])
A1 = np.array([[1,1,1],[1,-1,-1],[-1,-1,1],[-1,1,-1]])*1/np.sqrt(3)
A2 = np.array([[1,1,1,-1/np.sqrt(5)],[1,-1,-1,-1/np.sqrt(5)],[-1,1,-1,-1/np.sqrt(5)],[-1,-1,1,-1/np.sqrt(5)],[0,0,0,4/np.sqrt(5)]])
print(angle(A1[1,:],A1[2,:])*180/np.pi)
#%%
repeats = 100
angles = np.pi/180*np.array([0,5,6,7,8,9,10,14,15,20,30])
success_sim = np.zeros(len(angles))
for k, beta in enumerate(angles):
    for i in range(repeats):
        x0 = np.random.normal(0,1,n)
        xhat = init_angle(x0, beta)

        b_f = abs(A1@x0)
        # Solve problem with simplex measurements
        x = cp.Variable(n, complex=False)
        prob = cp.Problem(cp.Maximize((inp(x,xhat))),[cp.norm(cp.multiply(A1 @ x,1/b_f), "inf") <= 1])
        prob.solve(verbose=False)
        xsol = x.value
        alpha = inp(x0,xsol)/(inp(xsol,xsol))
        xsol = alpha * xsol
        error = la.norm(x0-xsol)**2/la.norm(x0)**2
        if error <  1e-4:
            success_sim[k] += 1
        
#%%
plt.plot(angles[0:7]*180/np.pi, success_sim[0:7]*100/repeats,'r')

# %%
