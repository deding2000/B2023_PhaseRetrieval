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
