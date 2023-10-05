#%%
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def basis_pursuit(m,A,z_true):
    x_hat = (A @ z_true).T
    z = cp.Variable(m, complex=True)
    prob = cp.Problem(cp.Minimize(cp.norm(z, 1)),[A @ z == x_hat])
    prob.solve()
    sol = z.value
    z_norm = cp.norm(sol,p=2).value
    res = ((cp.norm(z_true - sol, p=2).value)**2)*(1/(z_norm)**2)
    return sol, res, prob.value 

def PhaseMax(n, A, x_hat,x_test):
    # Generate data.
    b = abs(A@x_test)
    # Define and solve the CVXPY problem.
    x = cp.Variable(n, complex=True)
    prob = cp.Problem(cp.Maximize(cp.real(x_hat.T@ x)),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])

    prob.solve()

    sol = x.value
    alpha = (np.conj(sol.T)@x_test)/abs((np.conj(sol.T)@x_test))
    sol = alpha * sol
    
    x_norm = cp.norm(sol,p=2).value
    #res = cp.multiply(cp.norm(sol - x_test, p=2).value,1/(x_norm))
    res = ((cp.norm(sol - x_test, p=2).value)**2)*(1/(x_norm)**2)

    return sol, res, prob.value

def testPhasemax():
    n = 10
    ratios = np.array([3,4,5,6,7,8,9,10,11,12])
    ms = ratios*n
    simulations = 100
    #m = np.array([10,20,30,40,50,60,70,80,90,100])
    sol = np.zeros(100)
    res = np.zeros(10)
    success = 1
    nbsucceded = np.zeros(len(ratios))
    probs = np.zeros(len(ratios))
#np.random.seed(2)

    # Theoritical probabilities
    angle = 0.001
    alpha = 1 - (2/np.pi)*angle
    for j in range(len(ratios)) :
        m = ms[j]
        A = np.fft.fft(np.eye(m))/np.sqrt(m)
        A = A[:,np.random.choice(A.shape[0], n, replace=False)]
        probs[j] = 1-np.exp(-(alpha*m-4*n)**2/(2*m))

        for i in range(simulations):
            # A_real = np.random.randn(m, n)
            # A_im = np.random.randn(m, n)
            # A = A_real + 1j*A_im
            #A = np.fft.fft(np.eye(n))
            x_test = np.random.randn(n) + 1j*np.random.randn(n)
            x_test = x_test / np.linalg.norm(x_test)
            x_re,x_im = pol2cart(1, angle)
            x_hat = (x_re + 1j*x_im)*x_test
            # omega = (np.conj(x_hat.T)@ x_test) / np.abs((np.conj(x_hat.T) @ x_test))
            # x_hat = x_hat * omega / np.linalg.norm(x_hat)
            #x_hat = x_test
            # angle = np.arccos(np.real((np.conj(x_hat.T) @ x_test))/(np.linalg.norm(x_hat)*np.linalg.norm(x_test)))
            
            sol, res, prob = PhaseMax(n, A, x_hat,x_test)
            if res < success:
                nbsucceded[j] += 1

    plt.plot(ratios, nbsucceded*100/simulations)   
    plt.show() 

    #plt.plot(ratios, probs*100)   
    #plt.show()

testPhasemax()
# small test
# n = 10
# m = 80
# A = np.fft.fft(np.eye(m))/np.sqrt(m)
# A = A[:,np.random.choice(A.shape[0], n, replace=False)]
# x_test = np.random.randn(n) + 1j*np.random.randn(n)
# x_test = x_test / np.linalg.norm(x_test)
# x_re,x_im = pol2cart(1, 0.001)
# x_hat = (x_re + 1j*x_im)*x_test
# sol, res, prob = PhaseMax(n, A, x_hat,x_test)

# plt.plot(np.imag(sol), 'r')
# plt.plot(np.imag(x_test), 'b')
# plt.show()
# plt.plot(np.real(sol), 'r')
# plt.plot(np.real(x_test), 'b')
# plt.show()
# print(res)

def basis_pursuit_test():
    n = 10
    ratios = np.array([1,1.1,1.2,1.3,1.4,1.5])
    ms = (ratios*n).astype(int)
    simulations = 50
    #m = np.array([10,20,30,40,50,60,70,80,90,100])
    sol = np.zeros(100)
    res = np.zeros(10)
    success = 0.01
    nbsucceded = np.zeros(len(ratios))
    np.random.seed(1)
    for j in range(len(ratios)):
        m = ms[j]
        for i in range(simulations):
            A_real = np.random.randn(n,m)
            A_im = np.random.randn(n,m)
            A = A_real + 1j*A_im
            #A = np.matrix(A)
            #A = np.fft.fft(np.eye(n))
            x_test_real = np.random.randn(m)
            x_test_img = np.random.randn(m)
            x_test = x_test_real + 1j*x_test_img
            sol, res, prob = basis_pursuit(m, A,x_test)
            if res < success:
                nbsucceded[j] += 1

    plt.plot(ratios, nbsucceded*100/simulations)   
    plt.show() 


# %%
