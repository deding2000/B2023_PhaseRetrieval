import numpy as np
from fastapy.fastapy import Fasta
import matplotlib.pyplot as plt

def solvePhaseMax(A,b0,x0): #Not At and opts
    m = len(b0)
    n = len(x0)
    #remainIters 

    x0 =  (x0/np.linalg.norm(x0))*np.mean(b0)*(m/n)*100

    #  re-scale the initial guess so that it approximately satisfies |Ax|=b
    sol = x0 * min(b0 / abs(A(x0)))

    # Initialize values potentially computed at each round.
    # ending = 0;       % Indicate whether any of the ending condition (except maxIters) has been met in FASTA.
    # iter = 0;
    # currentTime = [];
    # currentResid = [];
    # currentReconError = [];
    # currentMeasurementError = [];

    # % Initialize vectors for recording convergence information
    # [solveTimes,measurementErrors,reconErrors,residuals] = initializeContainers(opts);

    ## Define objective function components for the gradient descent method FASTA
    
    def f(x):
        temp = A(x)-b0
        temp[temp<0] = 0
        return 0.5*np.linalg.norm(temp)**2
    

    def gradient(x):
        temp = A(x)-b0
        temp[temp<0] = 0
        return (np.sign(A(x))*temp) # max(abs(A(x))-b0,0))
   
    temp = A(x0)-b0
    temp[temp<0] = 0
    constraintError = np.linalg.norm(temp)
    Iters = 0
    while Iters < 10:   
        def proxg(x,t):
            return x+t*x0
        def g(x):
            return -np.real(x0.T*x)          
        lsq = Fasta(f, g, gradient, proxg)
        lsq.learn(x0)
        sol = lsq.coefs_

        x0 = x0/10                             
       
        # remainIters = remainIters - fastaOuts.iterationCount;
        # fastaOpts.maxIters = min(opts.maxIters, remainIters);
        temp = A(sol)-b0
        temp[temp<0] = 0
        newConstraintError = np.linalg.norm(temp)
        relativeChange = abs(constraintError-newConstraintError)/np.linalg.norm(b0)
        if relativeChange < 0.001: # opts.tol:
            break
        constraintError = newConstraintError
        Iters += 1
        
    return sol, constraintError

n = 10
m = 10
x_test = np.random.rand(n,1)
A = np.random.rand(m,n)
b = abs(A@x_test)
x0 = np.random.rand(n,1)
def Af(x):
    return A@x
sol, constraint = solvePhaseMax(Af, b, x0)

plt.plot(x_test, 'r')
plt.plot(sol, 'b')
plt.show()