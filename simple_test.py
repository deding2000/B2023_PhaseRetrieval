# %%
import numpy as np
from numpy import linalg as la
from Helper_functions import *

n = 5
m = 20
Data = GaussData(n,m,isComplex=False)
# Phaselift
xsol_lift= PhaseLift(Data.A, Data.b,verbose=False,isComplex=False)
#
sol = xsol_lift
X0 = np.outer(Data.x0,Data.x0)
error = la.norm(X0-sol)**2/la.norm(X0)**2
print(error)
#
U, S, Vh = np.linalg.svd(xsol_lift, full_matrices=False,hermitian=True)
xsol_lift = Vh[0,:]*np.sqrt(S[0])
alpha = inp(Data.x0,xsol_lift)/(inp(xsol_lift,xsol_lift))
sol = alpha * xsol_lift
error2 = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
print(error2)

#%% Phasemax
xhat = init_angle(Data.x0, np.pi/180*np.array([25]))
xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False,isComplex=True)
alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
sol = alpha * xsol
error2 = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
print(error2)

# %%
