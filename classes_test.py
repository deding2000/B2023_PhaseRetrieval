# GaussData and example of solver
# %%
import numpy as np
from numpy import linalg as la
from Helper_functions import *

np.random.seed(2)
class GaussData:
  def __init__(self, n, m, isComplex):
    self.x0 = np.random.randn(n) + isComplex*1j*np.random.randn(n)
    self.A = np.zeros((m,n),dtype = 'complex_')
    for i in range(n):
        self.A[:,i] = np.random.rand(m) + isComplex*1j*np.random.randn(m)
        self.A[:,i] = self.A[:,i] / la.norm(self.A[:,i])
    self.b = abs(self.A@self.x0)

# %%
# solve problem
# constants
n = 100
m = [200,300,400,500,600,800,1000]
angles = np.pi/180*np.array([25,36,45])
success = 1e-5
repeats = 10
nbsucceded = np.zeros((len(m), len(angles)))
for k, angle in enumerate(angles):
    for i, m1 in enumerate(m):
        for j in range(repeats):
            Data = GaussData(n,m1,True)
            xhat = init_angle(Data.x0, angle)
            xsol = PhaseMax(Data.A, Data.b, xhat)
            alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
            sol = alpha * xsol
            error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
            if error < success:
                        nbsucceded[i,k] += 1
#%%
plt.plot(m, nbsucceded[:,0]*100/repeats,'r')
plt.plot(m, nbsucceded[:,1]*100/repeats,'g')   
plt.plot(m, nbsucceded[:,2]*100/repeats,'b')       
plt.show() 


# %%
