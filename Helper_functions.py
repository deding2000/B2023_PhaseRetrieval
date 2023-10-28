import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

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

def PhaseMax(A, b, xhat):
    # Define and solve the CVXPY problem.
    n = A.shape[1]
    m = A.shape[0]
    x = cp.Variable(n, complex=True)
    prob = cp.Problem(cp.Maximize(cp.real(inp(x,xhat))),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])
    prob.solve()
    return x.value

def basis_pursuit(m,A,xhat):
    z = cp.Variable(m, complex=True)
    prob = cp.Problem(cp.Minimize(cp.norm(z, 1)),[A @ z == xhat])
    prob.solve()
    sol = z.value
    dual_sol = prob.constraints[0].dual_value
    return sol, dual_sol 