import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

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

def init_angle(xtrue, theta):
    n = len(xtrue)
    d = np.random.rand(n)

    d = d - (d.T @ xtrue)/np.linalg.norm(xtrue)**2 * xtrue
    d = d/np.linalg.norm(d) * np.linalg.norm(xtrue)

    xhat = xtrue + d * np.tan(theta)
    xhat = xhat/np.linalg.norm(xhat)*np.linalg.norm(xtrue)

    return xhat

def PhaseMax(A, b, xhat,verbose):
    # Define and solve the CVXPY problem.
    n = A.shape[1]
    m = A.shape[0]
    x = cp.Variable(n, complex=True)
    prob = cp.Problem(cp.Maximize(cp.real(inp(x,xhat))),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])
    prob.solve(verbose=verbose)
    return x.value

def basis_pursuit(m,A,xhat):
    z = cp.Variable(m, complex=True)
    prob = cp.Problem(cp.Minimize(cp.norm(z, 1)),[A @ z == xhat])
    prob.solve()
    sol = z.value
    dual_sol = prob.constraints[0].dual_value
    return sol, dual_sol 

def pcover1(m,n,angle):
   alpha = 1- (2/np.pi)*angle
   if 4*n < m*alpha:
        return (1 - np.exp(-((alpha*m-4*n)**2)*1/(2*m)))
   else:
        return 0

class GaussData:
  def __init__(self, n, m, isComplex):
    self.x0 = np.random.randn(n) + isComplex*1j*np.random.randn(n)
    self.A = np.zeros((m,n),dtype = 'complex_')
    for i in range(n):
        self.A[:,i] = np.random.rand(m) + isComplex*1j*np.random.randn(m)
        self.A[:,i] = self.A[:,i] / la.norm(self.A[:,i])
    self.b = abs(self.A@self.x0)

