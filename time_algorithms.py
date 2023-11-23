import time
import numpy as np
from numpy import linalg as la
from Helper_functions import *

n = 150
m = 900
angles = np.pi/180*np.array([25,36,45])
Data = GaussData(n,m,True)
B = np.diag((Data.b))
B1 = np.diag(1/(Data.b))
D = np.conj((Data.A).T)@B1
xhat = init_angle(Data.x0, angles[0])

################# TIMING ######################################33
start_time = time.time()
xsol = PhaseMax(Data.A, Data.b, xhat,isComplex =True, verbose=False)
alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
sol = alpha * xsol
print("--- %s PM seconds ---" % (time.time() - start_time))

start_time = time.time()
zsol = basis_pursuit(m,D,xhat,verbose=False,isComplex=True)
zb = zsol/abs(zsol)
xsol = la.lstsq(Data.A,B@zb,rcond=None)[0]
alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
sol = alpha * xsol
print("--- %s BP seconds ---" % (time.time() - start_time))

start_time = time.time()
xsol = PhaseLift(Data.A, Data.b, xhat,isComplex =True, verbose=False)
alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
sol = alpha * xsol
print("--- %s PL seconds ---" % (time.time() - start_time))


