import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
#import clarabel

def save_points(x,y,path):
    data = np.zeros((len(x),2))
    for i in range(len(x)):
      data[i,0] = x[i]
      data[i,1] = y[i]
    np.savetxt(path, data) 

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def inp(x,y):
    return x.T @ np.conjugate(y)

def angle(x,y):
    return np.arccos(np.real(inp(x,y))/(la.norm(x)*la.norm(y)))

def init_angle(xtrue, theta):
    n = len(xtrue)
    d = np.random.normal(0,1,n)

    d = d - (d.T @ xtrue)/np.linalg.norm(xtrue)**2 * xtrue
    d = d/np.linalg.norm(d) * np.linalg.norm(xtrue)

    xhat = xtrue + d * np.tan(theta)
    xhat = xhat/np.linalg.norm(xhat)*np.linalg.norm(xtrue)

    return xhat

def PhaseMax(A, b, xhat,verbose, isComplex):
    # Define and solve the CVXPY problem.
    n = A.shape[1]
    m = A.shape[0]
    x = cp.Variable(n, complex=isComplex)
    prob = cp.Problem(cp.Maximize(cp.real(inp(x,xhat))),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])
    prob.solve(verbose=verbose,solver="ECOS")
    return x.value

def basis_pursuit(m,D,xhat,verbose,isComplex):
    z = cp.Variable(m, complex=isComplex)
    prob = cp.Problem(cp.Minimize(cp.norm(z, 1)),[D @ z == xhat])
    prob.solve(verbose=verbose)
    #dual_sol = prob.constraints[0].dual_value
    return z.value

def PhaseLift(A,b,verbose, isComplex):
    n = A.shape[1]
    m = A.shape[0]
    if isComplex:
        X = cp.Variable((n,n), hermitian=True)
    else:
        X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [
    inp(X @ A[i,:],A[i,:]) == b[i]**2 for i in range(m) ]
    prob = cp.Problem(cp.Minimize(cp.trace(X)),constraints)
    prob.solve(verbose=verbose)
    if isComplex:
        U, S, Vh = np.linalg.svd(X.value, full_matrices=False,hermitian=True)
    else:
         U, S, Vh = np.linalg.svd(X.value, full_matrices=False,symmetric=True)
    sol = Vh[0,:]*np.sqrt(S[0])
    return sol

def pcover1(m,n,angle):
   alpha = 1- (2/np.pi)*angle
   if 4*n < m*alpha:
        return (1 - np.exp(-((alpha*m-4*n)**2)*1/(2*m)))
   else:
        return 0

class GaussData:
  def __init__(self, n, m, isComplex):
    self.x0 = np.random.normal(0,1,n) + isComplex*1j*np.random.normal(0,1,n)
    self.A = np.zeros((m,n),dtype = 'complex_')
    for i in range(n):
        self.A[:,i] = np.random.normal(0,1,m) + isComplex*1j*np.random.normal(0,1,m)
        self.A[:,i] = self.A[:,i] / la.norm(self.A[:,i])
    self.b = abs(self.A@self.x0)
