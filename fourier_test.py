#%%

import numpy as np
from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp
import itertools 
import math

def find_angle(A):
    m,n = A.shape
    angles = np.zeros((m,m))
    for i in range(m):
        for j in range(i+1,m):
            angles[i,j] = angle(A[i,:],A[j,:])*180/np.pi
    angles = angles[angles != 0]
    return  angles.tolist()

def test_fourier(isComplex = False, m = 4, n = 2, angles = [np.pi/180*25, np.pi/180*36, np.pi/180*45], repeats = 1000, number_combinations = None):
    """
    Only suitable for small m and n. Goes through all possible combinations of n columns 
    from FFT (or number_combinations if specified) and tests success rate for each angle 
    in angles. Repeats are done with different x0. 

    n: dimension of x
    m: meassurements
    angles: list of angles to test
    repeats: number of simulations 
    number_combinations: number of samples from FFT, if None all combinations are used
    """
    
    # create FFT matrix
    FFT = np.fft.fft(np.eye(m))
    
    # dictionary to store results
    results = {"columns": [], "angle" : angles, "success": [], "anglesA": []}

    # all combinations of n columns from FFT
    combinations = itertools.combinations(range(m), n)

    if number_combinations is not None:
        # random array with number_combinations 1's 
        k = math.comb(m,n)
        temp = np.array([0]*(k-number_combinations) + [1]*number_combinations)
        np.random.shuffle(temp)

        # keep only combinations with 1's
        combinations = [comb for comb, condition in zip(combinations, temp) if condition]
    for combination in combinations:
        A = FFT[:,combination]
        
        results["columns"].append(combination)

        results["anglesA"].append(find_angle(A))

        # store a success rate for each angle
        successes = []
        for beta in angles:
            # success rate for this angle
            success = 0
            for _ in range(repeats):
                # making test problem
                x0 = np.random.normal(0,1,n) + isComplex*1j*np.random.normal(0,1,n)
                b = abs(A@x0)
                
                # construct xhat from beta and x0
                xhat = init_angle(x0, beta)

                #solving
                x = cp.Variable(n, complex=isComplex)
                prob = cp.Problem((cp.Maximize(cp.real(inp(x,xhat)))),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])
                prob.solve(verbose=False,solver="ECOS")
                xsol = x.value
                alpha = inp(x0,xsol)/inp(xsol,xsol)
                xsol = alpha * xsol
                error = la.norm(x0-xsol)**2/la.norm(x0)**2
                if error <  1e-5:
                    success += 1   
            # save success rate for this angle
            successes.append(success/repeats)
        results["success"].append(successes)    
    return results

#%%
results8c = test_fourier(m=8,n=2,number_combinations=1000, isComplex=True)
results6c = test_fourier(m=6,n=2,number_combinations=1000, isComplex=True)

#%%
plt.plot(results8c["success"])
#%%
plt.plot(results6c["success"])
print(results6c["columns"][2],results6c["columns"][7],results6c["columns"][11],results6c["columns"][14]) 
F = np.fft.fft(np.eye(6))
A1 = F[:,[(0,3)]]
A1 = np.squeeze(A1, axis=1)
A2 = np.squeeze(F[:,[(1,4)]], axis=1)
A3 = np.squeeze(F[:,[(2,5)]], axis=1)
A4 = np.squeeze(F[:,[(4,5)]], axis=1)
print(find_angle(A1))
print(find_angle(A2))
print(find_angle(A3)) 
print(find_angle(A4))
# all have 90 or 120

print("good angles")
B1 = np.squeeze(F[:,[(0,1)]], axis=1)
B2 = np.squeeze(F[:,[(0,2)]], axis=1)
B3 = np.squeeze(F[:,[(0,4)]], axis=1)
B4 = np.squeeze(F[:,[(0,5)]], axis=1)
B5 = np.squeeze(F[:,[(1,2)]], axis=1)
print(find_angle(B1))
print(find_angle(B2))
print(find_angle(B3))
print(find_angle(B4))
# variaty of angles, 41, 75, 90

#%%
results8 = test_fourier(m=8,n=2)
import json
with open('output8.json', 'w') as json_file:
    json.dump(results8, json_file)

#%%
results2 = test_fourier(m = 2)
results4 = test_fourier(m = 4)
results6 = test_fourier(m = 6)


#%%
import json

with open('output.json', 'w') as json_file:
    json.dump(results2, json_file)
    json.dump(results4, json_file)
    json.dump(results6, json_file)


#%%

n = 10
#m = np.array([200,300,400,500,600,700,800])
m = 20
number_combinations = 10
repeats = 100
#for m1 in m:
results = test_fourier(m = m, n = n, repeats = repeats, number_combinations = number_combinations)
#results["success"]
   # with open(f'output_m{m1}.json', 'w') as json_file:
   #     json.dump(results, json_file)

#%%
plt.plot(results4["success"])
#%% 
results6 = test_fourier(m = 6)
#%%
fig, ax = plt.subplots(1,2)
ax[0].plot(results4["success"], label = results4["angle"])
ax[0].legend()
ax[1].plot(results6["success"], label = results6["angle"])
ax[1].legend()
plt.show()

print(f"Average success rate for m = 2: {np.mean(results2['success'])}")
print(f"Average success rate for m = 4: {np.mean(results4['success'])}")
print(f"Average success rate for m = 6: {np.mean(results6['success'])}")

