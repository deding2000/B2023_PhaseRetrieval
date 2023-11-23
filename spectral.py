#%%
from scipy import linalg as la
import numpy as np
from Helper_functions import *

def spectral_initializer(A,b,n,m, truncate = True, isScaled = False, optimal = False):
    mu = np.mean(b**2)
    if truncate:
        mu = np.mean(b**2)
        b0 = b**2 <= 3**2 * mu
        # for i in range(m):
        #     if b[i]**2 > 3**2 * mu:
        #         b0[i] = 0
        print(m - np.sum(b0))
        b = b*b0
    temp = 1/m* np.conj(A.T) @ np.diag(b**2)@ A
    w, vr = la.eig(temp)
    largest_eigenvector = vr[:, np.argmax(w)]
    if isScaled:
        Ax = np.abs(A @ largest_eigenvector)*b0
        u = Ax * b
        l = Ax * Ax
        s = la.norm(u)/la.norm(l)
        largest_eigenvector = largest_eigenvector * s
    if optimal:
        delta = m/n
        y = b**2/mu
        ymax = np.max(y,0)
        print(ymax)
        T = (ymax-1)/(ymax+np.sqrt(delta)-1)
        temp = 1/m * T* np.conj(A.T) @  A
    b0 = np.ones(m)
    
    return largest_eigenvector

ns = [5]
m = 2
error = np.zeros(len(ns))
error_trunc = np.zeros(len(ns))
error_opt = np.zeros(len(ns))
for i,n in enumerate(ns):
    m = 6*n
    Data = GaussData(n,m,True)
    Data.b += np.random.normal(0,0.1,m)
    xhat_trunc = spectral_initializer(Data.A,Data.b,n,m, truncate=True, isScaled=False, optimal = False)
    xhat = spectral_initializer(Data.A,Data.b,n,m, truncate=False, isScaled=False, optimal = False)
    xhat_opt = spectral_initializer(Data.A,Data.b,n,m, truncate=False, isScaled=False, optimal = True)
    error[i] = la.norm(Data.x0-xhat)**2/la.norm(Data.x0)**2 
    error_trunc[i] = la.norm(Data.x0-xhat_trunc)**2/la.norm(Data.x0)**2 
    error_opt[i] = la.norm(Data.x0-xhat_opt)**2/la.norm(Data.x0)**2   
print(f"error: {error}, error_trunc: {error_trunc}")
plt.plot(ns,error)
plt.plot(ns,error_trunc)
plt.plot(ns,error_opt)
plt.legend(["error", "error_trunc", "error_opt"])
plt.xlabel("n")
plt.ylabel("error")
plt.show()


#%%
n = 100
#n = 10
#ms = [60]
ms = [600,700]
#m = 60
success_specctral = np.zeros(len(ms))
success_specctral2 = np.zeros(len(ms))
success_random = np.zeros(len(ms))
success_angle = np.zeros(len(ms))

repeats = 10

for i,m in enumerate(ms):
    print(m)
    for _ in range(repeats):
        Data = GaussData(n,m,True)
        xhat = spectral_initializer(Data.A,Data.b,n,m, truncate=True)
        #print(angle(xhat,Data.x0)*180/np.pi)
        xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False, isComplex=True)
        alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
        sol = alpha * xsol
        error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
        if error < 1e-5:
            success_specctral[i] +=1

        xhat = spectral_initializer(Data.A,Data.b,n,m, truncate=False)
        #print(angle(xhat,Data.x0)*180/np.pi)
        xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False, isComplex=True)
        alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
        sol = alpha * xsol
        error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
        if error < 1e-5:
            success_specctral2[i] +=1
        
        xhat = np.random.normal(0,1,n) + 1j*np.random.normal(0,1,n)
        #print(angle(xhat,Data.x0)*180/np.pi)
        xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False, isComplex=True)
        alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
        sol = alpha * xsol
        error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
        if error < 1e-5:
            success_random[i] +=1  

        xhat = init_angle(Data.x0, np.pi/180*np.array([25]))
        xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False, isComplex=True)
        alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
        sol = alpha * xsol
        error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
        if error < 1e-5:
            success_angle[i] +=1 


# plt.plot(ms,success_random/repeats)
# plt.plot(ms,success_specctral/repeats)
# plt.plot(ms,success_angle/repeats)
# plt.legend(["random", "spectral", "angle"])
# plt.xlabel("m")
# plt.ylabel("success rate")
# plt.show()
# #xhat = np.random.normal(0,1,n) + 1j*np.random.normal(0,1,n)
# #xhat = init_angle(Data.x0, np.pi/180*np.array([25]))
# # xsol = PhaseMax(Data.A, Data.b, xhat,verbose=False, isComplex=True)
# # alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
# # sol = alpha * xsol
# # error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
print(f"success_random: {success_random}, success_spectral_trunc: {success_specctral}, successs_spectral: {success_specctral2}, success_angle: {success_angle}")
