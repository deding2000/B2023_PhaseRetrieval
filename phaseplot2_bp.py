# GaussData with different angles
# %%
import numpy as np
from numpy import linalg as la
from Helper_functions import *
#import clarabel

# solve problem
# constants
np.random.seed(70)
n = 100
m = np.array([200,300,400,450,500,550,600,800,900,1000])
angles = np.pi/180*np.array([25,36,45])
success = 1e-5
repeats = 20
nbsucceded = np.zeros((len(m), len(angles)))

#%%
for k, beta in enumerate(angles):
    for i, m1 in enumerate(m):
        Data = GaussData(n,m1,True)
        B = np.diag((Data.b))
        B1 = np.diag(1/(Data.b))
        D = np.conj((Data.A).T)@B1
        for j in range(repeats):
            xhat = init_angle(Data.x0, beta)
            zsol = basis_pursuit(m1,D,xhat,verbose=False,isComplex=True)
            zb = zsol/abs(zsol)
            xsol = la.lstsq(Data.A,B@zb,rcond=None)[0]
            alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
            sol = alpha * xsol
            error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
            if error < success:
                        nbsucceded[i,k] += 1

plt.plot(m, nbsucceded[:,0]*100/repeats,'r')
plt.plot(m, nbsucceded[:,1]*100/repeats,'g')  
plt.plot(m, nbsucceded[:,2]*100/repeats,'b')
plt.show() 

# %%
