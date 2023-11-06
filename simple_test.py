# %%
import numpy as np
from numpy import linalg as la
from Helper_functions import *

n = 10
m = 60
Data = GaussData(n,m,True)
xhat = init_angle(Data.x0, np.pi/180*np.array([25]))
xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False)
alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
sol = alpha * xsol
error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
print(error)


# %%
# %%
