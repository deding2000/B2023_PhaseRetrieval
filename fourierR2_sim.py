#%%
import numpy as np
from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp
# Real and simple case with different angles
n = 2
m = 2
A = np.fft.fft(np.eye(m))
A = A[:,np.random.choice(A.shape[0], n, replace=False)]
repeats = 1000
angles = np.pi/180*np.array([0,10,20,30,40,45,50,60,70,80,90])
success_sim = np.zeros(len(angles))
success_true = np.zeros(len(angles))
success_uni = np.zeros(len(angles))
for k, beta in enumerate(angles):
    success_true[k] = 1-2*beta/np.pi
    DGauss = GaussData(n,m,False)
    A_unif = DGauss.A
    for i in range(repeats):
        x0 = np.random.normal(0,1,n)
        xhat = init_angle(x0, beta)
        b_f = abs(A@x0)
        b_uni = abs(A_unif@x0)
        # Solve fourier problem
        x = cp.Variable(n, complex=False)
        prob = cp.Problem(cp.Maximize(cp.real(inp(x,xhat))),[cp.norm(cp.multiply(A @ x,1/b_f), "inf") <= 1])
        prob.solve(verbose=False,solver="ECOS")
        xsol = x.value
        alpha = inp(x0,xsol)/(inp(xsol,xsol))
        xsol = alpha * xsol
        error = la.norm(x0-xsol)**2/la.norm(x0)**2
        if error <  1e-5:
            success_sim[k] += 1
        # Solve uniform problem
        # x2 = cp.Variable(n, complex=False)
        # prob2 = cp.Problem(cp.Maximize(cp.real(inp(x2,xhat))),[cp.norm(cp.multiply(A_unif @ x2,1/b_uni), "inf") <= 1])
        # prob2.solve(verbose=False,solver="ECOS")
        # xsol2 = x2.value
        # alpha2 = inp(x0,xsol2)/(inp(xsol2,xsol2))
        # xsol2 = alpha2 * xsol2
        # error2 = la.norm(x0-xsol2)**2/la.norm(x0)**2
        # if error <  1e-5:
        #     success_uni[k] += 1
        

plt.plot(angles*180/np.pi, success_sim*100/repeats,'r')
plt.plot(angles*180/np.pi, success_true*100,'b')
#%%
save_points(angles*180/np.pi,success_sim*100/repeats,'Data/fourierR2_sim1000.txt')
save_points(angles*180/np.pi,success_true*100,'Data/fourierR2_true.txt')
#plt.plot(angles*180/np.pi, success_uni*100/repeats,'g')



# %%
