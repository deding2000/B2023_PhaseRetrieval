# GaussData with different angles
# %%
import numpy as np
from numpy import linalg as la
from Helper_functions import *
#import clarabel

# solve problem
# constants
np.random.seed(65)
n = 10
m = np.array([300,400,450,500,550,600,800,900,1000])
angles = np.pi/180*np.array([25])#,36,45])
success = 1e-5
repeats = 10
nbsucceded = np.zeros((len(m), len(angles)))

#%%
for k, beta in enumerate(angles):
    for i, m1 in enumerate(m):
        Data = GaussData(n,m1,True)
        for j in range(repeats):
            xhat = init_angle(Data.x0, beta)
            xsol = PhaseMax(Data.A, Data.b, xhat,isComplex =True, verbose=False)
            alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
            sol = alpha * xsol
            error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
            if error < success:
                        nbsucceded[i,k] += 1

plt.plot(m, nbsucceded[:,0]*100/repeats,'r')
# plt.plot(m, nbsucceded[:,1]*100/repeats,'g')  
# plt.plot(m, nbsucceded[:,2]*100/repeats,'b')
plt.show() 
#%%
################################ Save Data ####################################
# x = np.array(m)
# beta25 = np.array(nbsucceded[:,0]*100/repeats)
# save_points(x,beta25,'Data/beta25_n100.txt')
# beta36 = np.array(nbsucceded[:,1]*100/repeats)
# save_points(x,beta36,'Data/beta36_n100.txt')
# beta45 = np.array(nbsucceded[:,2]*100/repeats)
# save_points(x,beta45,'Data/beta45_n100.txt')

# # %%
# # Theoretical p_cover 
# mprob = np.array([200,300,400,450,500,550,600,650,700,800,850,900,1000,1100,1200])
# xprob = np.array(mprob)
# probs = np.zeros((len(mprob)))
# betaprob = (np.pi/180)*25
# for i,m1 in enumerate(mprob):
#         # Theoretical p_cover 
#         probs[i] = pcover1(m1,n,betaprob)
# plt.plot(mprob, probs*100,'r--')  
# save_points(mprob,probs*100,'Data/prob25_n100.txt')

