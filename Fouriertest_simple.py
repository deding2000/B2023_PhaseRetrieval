#%%
import numpy as np
from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp
import math

def find_angle(A):
    m,n = A.shape
    angles = np.zeros((m,m))
    for i in range(m):
        for j in range(i+1,m):
            angles[i,j] = angle(A[i,:],A[j,:])*180/np.pi
    angles = angles[angles != 0]
    return np.median(angles) # angles.tolist()

def fourier_test_random(n = 10, m = 20, angles = [np.pi/8, np.pi/6, np.pi/4], repeats = 1000, sample_A = 10):
    """
    For each angle in angles, test success rate for sample_A different randomly generated A from FFT, 
    for each randomly genreated A, repeat are done with different x0

    n: dimension of x
    m: meassurements
    angles: list of angles to test
    repeats: number of simulations
    sample_A: number of different samples from FFT
    returns: dictionary with results
    """
    FFT = np.fft.fft(np.eye(m))

    # dictionary to store results
    results = {"angle":[], "success":[], "anglesA":[] }

    # iterate over angles
    for beta in angles:
        # save angle to results
        results["angle"].append(beta)

        # to store angles between rows of A
        anglesA = []

        # to store success rate
        successes = []
        for j in range(sample_A):
            
            # samle n columns from FFT
            A = FFT[:,np.random.choice(FFT.shape[0], n, replace=False)]
            
            # save angles between rows of A
            anglesA.append(find_angle(A))
        
            # to store number of successes for this sample of A
            success = 0
            for i in range(repeats):
                # making test problem
                x0 = np.random.normal(0,1,n)
                b = abs(A@x0)
                
                # construct xhat from beta and x0
                xhat = init_angle(x0, beta)

                #solving
                x = cp.Variable(n, complex=False)
                prob = cp.Problem((cp.Maximize(x.T@xhat)),[cp.norm((A @ x*1/b), "inf") <= 1])
                prob.solve(verbose=False,solver="ECOS")
                xsol = x.value
                alpha = (x0.T@xsol)/(xsol.T@xsol.T)
                xsol = alpha * xsol
                error = la.norm(x0-xsol)**2/la.norm(x0)**2
                if error <  1e-5:
                    success += 1
            
            # save success rate for this sample of A
            successes.append(success/repeats)
        
        # store the success rate for this angle
        results["success"].append(successes)
        # store the angles between rows of A for this angle
        results["anglesA"].append(anglesA)
    return results

n = 100
ms = np.array([200, 300, 400, 500, 600, 700])
for m in ms:
    result = fourier_test_random(n = n, m = m)
    plt.figure()
    plt.plot(result["success"])
    plt.title("m = " + str(m))
    plt.show()

#%%
# plot results
fig, ax = plt.subplots(1,3, figsize=(15,5))

# beta = pi/8
for i, beta in enumerate(results["angle"]):
    ax[i].plot(results["anglesA"][i], results["success"][i], 'o') #, label = 'beta = ' + str(results["angle"][i]*180/np.pi))
    ax[i].set_xlabel('Angle between rows of A')
    ax[i].set_ylabel('Success rate')
    ax[i].set_title('beta = ' + str(np.round(results["angle"][i]*180/np.pi)))
plt.show()

plt.figure()
for i,_ in enumerate(results["angle"]):
    plt.plot(results["success"][i])
plt.show()
#%%

# set dimensions
n = 2
m = 4

FFT = np.fft.fft(np.eye(m))

# test angles
angles = [np.pi/8, np.pi/6, np.pi/4]

# number of possible samples from FFT
sample_number = math.comb(m,n)

# number of repeats
repeats = 1000

# dictionary to store results
results = {"columns": [], "angle" : angles, "success": [], "anglesA": []}
#results = {"angle":[], "success":[], "anglesA":[] }


for l in range(m):
    for j in range(l+1,m):
        A = FFT[:,[l,j]]
        
        results["columns"].append([l,j])

        #TODO angles of A

        # store a success rate for each angle
        successes = []
        for beta in angles:
           # success rate for this angle
            success = 0
            for i in range(repeats):
                # making test problem
                x0 = np.random.normal(0,1,n)
                b = abs(A@x0)
                
                # construct xhat from beta and x0
                xhat = init_angle(x0, beta)

                #solving
                x = cp.Variable(n, complex=False)
                prob = cp.Problem((cp.Maximize(x.T@xhat)),[cp.norm((A @ x*1/b), "inf") <= 1])
                prob.solve(verbose=False,solver="ECOS")
                xsol = x.value
                alpha = (x0.T@xsol)/(xsol.T@xsol.T)
                xsol = alpha * xsol
                error = la.norm(x0-xsol)**2/la.norm(x0)**2
                if error <  1e-5:
                    success += 1   
            # save success rate for this angle
            successes.append(success/repeats)
        results["success"].append(successes)    
print(results)
#%%
plt.figure()
plt.plot(results["success"], label = results["angle"])
plt.legend()
plt.show()



n = 2
m = 4
A = np.fft.fft(np.eye(m))
A = A[:,np.random.choice(A.shape[0], n, replace=False)]
A = np.array([[ 1,1],
 [ 1,-1j],
 [ 1,-1],
 [ 1,1j]])
print(A)
repeats = 2000
success = np.zeros(repeats)
for i in range(repeats):
    x0 = np.random.normal(0,1,n) + 1j*np.random.normal(0,1,n)
    b = abs(A@x0)
    xhat = np.random.normal(0,1,n)+1j*np.random.normal(0,1,n)
    x = cp.Variable(n, complex=True)
    prob = cp.Problem(cp.Maximize(cp.real(inp(x,xhat))),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])
    prob.solve(verbose=False,solver="ECOS")
    xsol = x.value
    alpha = inp(x0,xsol)/(inp(xsol,xsol))
    xsol = alpha * xsol
    error = la.norm(x0-xsol)**2/la.norm(x0)**2
    if error <  1e-5:
        success[i] += 1

print(np.sum(success)/repeats)
# %%
