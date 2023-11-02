# GaussData and example of solver
# %%
import numpy as np
from numpy import linalg as la
from Helper_functions import *

def pcover1(m,n,angle):
   alpha = 1- (2/np.pi)*angle
   if 4*n < m*alpha:
        return (1 - np.exp(-((alpha*m-4*n)**2)*1/(2*m)))
   else:
        return 0

# %%
# solve problem
# constants
np.random.seed(11)
n = 100
m = np.array([400,450,500,550,600,1000])
angles = np.pi/180*np.array([25])#,36,45])
success = 1e-5
repeats = 5
nbsucceded = np.zeros((len(m), len(angles)))
probs = np.zeros((len(m), len(angles)))
#%%
for k, angle in enumerate(angles):
    for i, m1 in enumerate(m):
        # Theoretical p_cover 

        probs[i,k] = pcover1(m1,n,angle)
        for j in range(repeats):
            Data = GaussData(n,m1,True)
            xhat = init_angle(Data.x0, angle)
            xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False,solver="ECOS")
            alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
            sol = alpha * xsol
            error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
            if m1 > 550:
                print(error)
            if error < success:
                        nbsucceded[i,k] += 1

        
#%%
plt.plot(m, nbsucceded[:,0]*100/repeats,'r')
plt.plot(m, probs[:,0]*100,'r--')  
# plt.plot(m, nbsucceded[:,1]*100/repeats,'g')
# plt.plot(m, probs[:,1]*100,'g--')     
# plt.plot(m, nbsucceded[:,2]*100/repeats,'b')
# plt.plot(m, probs[:,2]*100,'b--')    
 
plt.show() 
#%%
# Save data
x = np.array(m)
y = np.array(nbsucceded[:,0]*100/repeats)
save_points(x,y,'Data/beta25.txt')

