#%%
import time
import numpy as np
from numpy import linalg as la
from Helper_functions import *

ns = [100,150,200,250,300,350,400,450,500]
ns = np.array([100,150,200,250,300])#,400,500,600,700,800,900,1000])
repeats = 5
# time_pm = np.zeros((len(ns)))
# time_bp = np.zeros((len(ns)))
time_pl = np.zeros((len(ns)))
for i, n in enumerate(ns):
    m = 5*n
    Data = GaussData(n,m,True)
    B = np.diag((Data.b))
    B1 = np.diag(1/(Data.b))
    D = np.conj((Data.A).T)@B1
    xhat = init_angle(Data.x0, np.pi*10/180)
    for j in range(repeats):
        ################# TIMING ######################################
        # start_time = time.time()
        # xsol = PhaseMax(Data.A, Data.b, xhat,isComplex =True)
        # alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
        # sol = alpha * xsol
        # #print("--- %s PM seconds ---" % (time.time() - start_time))
        # time_pm[i] += time.time() - start_time

        # start_time = time.time()
        # zsol = basis_pursuit(m,D,xhat,isComplex=True)
        # zb = zsol/abs(zsol)
        # xsol = la.lstsq(Data.A,B@zb,rcond=None)[0]
        # alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
        # sol = alpha * xsol
        # #print("--- %s BP seconds ---" % (time.time() - start_time))
        # time_bp[i] += time.time() - start_time

        start_time = time.time()
        xsol = PhaseLift(Data.A, Data.b, xhat,isComplex =True)
        alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
        sol = alpha * xsol
        #print("--- %s PL seconds ---" % (time.time() - start_time))
        time_pl[i] += time.time() - start_time

plt.plot(ns, time_pm/repeats,'r')
plt.plot(ns, time_bp/repeats,'g')  
# plt.plot(ns, time_pl/repeats,'b')

# %%
save_points(ns,time_pm/repeats,'Data/PM_time.txt')
save_points(ns,time_bp/repeats,'Data/BP_time.txt')
#plt.plot(np.log(ns), (time_pm/repeats),'r')
#plt.plot(np.log(ns), (time_bp/repeats),'g')  
# %%
