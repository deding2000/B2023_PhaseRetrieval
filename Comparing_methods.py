from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp
# from spectral import spectral_initializer

def error_cal(xsol):
    alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
    sol = alpha * xsol
    error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
    return error

def spectral_initializer(A,b,n,m, truncate = True, isScaled = False, optimal = False):
    mu = np.mean(b**2)
    if truncate:
        mu = np.mean(b**2)
        b0 = b**2 <= 3**2 * mu
        # for i in range(m):
        #     if b[i]**2 > 3**2 * mu:
        #         b0[i] = 0
        # print(m - np.sum(b0))
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


ratios =  np.arange(1, 4, 0.25) 
n = 100 
repeats = 10
errors = {"PhaseMax": [], "PhaseLift": [], "BasisPursuit": [], "PhaseLamp": [], "PhaseCut":[]}
for ratio in ratios:
    errorPM = 0
    errorPL = 0
    errorBP = 0
    errorPC = 0
    errorPLamp = 0
    for i in range(repeats):
        m = np.round(ratio*n).astype(int)
        Data = GaussData(n,m,True)
        B = np.diag((Data.b))

        xhat = spectral_initializer(Data.A, Data.b,n,m,  truncate = True, isScaled = False, optimal = False)
       
        ### PHASEMAX
        x_PM = PhaseMax(Data.A, Data.b, xhat,isComplex =True, verbose=False)
        errorPM += error_cal(x_PM)

        ### PHASELIFT
        x_PL = PhaseLift(Data.A, Data.b,verbose=False, isComplex=True)
        errorPL += error_cal(x_PL)

        ### BASIS PURSUIT
        B1 = np.diag(1/(Data.b))
        D = np.conj((Data.A).T)@B1
        zsol = basis_pursuit(m,D,xhat,verbose=False, isComplex=True)
        zb = zsol/abs(zsol)
        xsol = la.lstsq(Data.A,B@zb,rcond=None)[0]
        alpha = inp(Data.x0,xsol)/(inp(xsol,xsol))
        sol = alpha * xsol
        error = la.norm(Data.x0-sol)**2/la.norm(Data.x0)**2
        errorBP += error

        ### PHASECUT
        x_PC= PhaseCut(Data.A, Data.b,verbose=False, isComplex=True)
        xsol = la.lstsq(Data.A,B@x_PC,rcond=None)[0]
        errorPC += error_cal(xsol)

        ### PHASELAMP
        x_PLamp = PhaseLamp(Data.A,Data.b,xhat, k=10, epsilon = 10e-4,verbose=False, isComplex=True)
        errorPLamp += error_cal(x_PLamp)

    errors["PhaseMax"].append(errorPM/repeats)
    errors["PhaseLift"].append(errorPL/repeats)
    errors["BasisPursuit"].append(errorBP/repeats)
    errors["PhaseCut"].append(errorPC/repeats)
    errors["PhaseLamp"].append(errorPLamp/repeats)


plt.plot(ratios, errors["PhaseMax"],'r')
plt.plot(ratios, errors["PhaseLift"],'g')
plt.plot(ratios, errors["BasisPursuit"],'b')
plt.plot(ratios, errors["PhaseCut"],'y')
plt.plot(ratios, errors["PhaseLamp"],'k')
plt.legend(["PhaseMax","PhaseLift","BasisPursuit","PhaseCut","PhaseLamp","baseline"])
plt.show()
