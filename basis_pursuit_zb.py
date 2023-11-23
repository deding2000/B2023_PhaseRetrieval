#%%
import numpy as np
from numpy import linalg as la
from Helper_functions import *

#%%
n = 15
m = 70
Data = GaussData(n,m,isComplex=True)

#Phasemax
xhat = init_angle(Data.x0, np.pi/180*np.array([25]))
xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False,isComplex=True)
alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
xsol_max = alpha * xsol
error2 = la.norm(Data.x0-xsol_max)**2/la.norm(Data.x0)**2
print(error2)

#Dual problem
B1 = np.diag(1/(Data.b))
D = np.conj((Data.A).T)@B1
zsol = basis_pursuit(m,D,xhat)
zb = np.zeros(m,dtype = 'complex_')
meas = np.zeros(m,dtype = 'complex_')
for i in range(m):
    zb[i] = (Data.b)[i]*(zsol[i]/abs(zsol[i]))
print(la.norm(zb-Data.A@xsol)**2/la.norm(Data.A@xsol)**2)
# %%
