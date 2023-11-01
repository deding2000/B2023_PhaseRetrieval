#%%
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def basis_pursuit(m,D,x_hat):
    z = cp.Variable(m, complex=True)
    prob = cp.Problem(cp.Minimize(cp.norm(z, 1)),[D @ z == x_hat])
    prob.solve()
    sol = z.value
    dual_sol = prob.constraints[0].dual_value
    return sol, dual_sol

def PhaseMax(n, A, x_hat,x_test):
    # Generate data.
    b = abs(A@x_test)
    # Define and solve the CVXPY problem.
    x = cp.Variable(n, complex=True)
    prob = cp.Problem(cp.Maximize(cp.real(np.conj(x_hat.T)@ x)),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])

    prob.solve()

    sol = x.value
    #alpha = ((sol.T)@x_test)/abs(((sol.T)@x_test)) # modulus 1
    #alpha = ((sol.T)@x_test)/((sol.T)@sol)
    #sol = alpha * sol
    #res = cp.multiply(cp.norm(sol - x_test, p=2).value,1/(x_norm))
    #res = ((np.linalg.norm(x_test - sol))**2)*(1/(x_norm)**2)
    #res = ((np.linalg.norm(x_test-sol)))/(np.linalg.norm(x_test))
    res = np.mean(abs(sol)-abs(x_test))
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

def init_angle(xtrue, theta):
    n = len(xtrue)
    d = np.random.rand(n)

    d = d - (d.T @ xtrue)/np.linalg.norm(xtrue)**2 * xtrue
    d = d/np.linalg.norm(d) * np.linalg.norm(xtrue)

    xhat = xtrue + d * np.tan(theta)
    xhat = xhat/np.linalg.norm(xhat)*np.linalg.norm(xtrue)

    return xhat

def gauss1D(iscomplex, n, m):
    xtrue = np.random.randn(n) + iscomplex * 1j * np.random.randn(n)

    A = np.random.rand(m,n) + iscomplex * 1j * np.random.rand(m,n)

    return A, xtrue

def gauss_phaseplot(iscomplex, num_trials, n, theta):
    
    ratio = np.array([1,3,5,7,9,11])
    num_ratio = len(ratio)

    results = np.zeros((num_ratio, num_trials))

    for p in range(num_ratio):

        for q in range(num_trials):
            m = np.round(ratio[p]*n).astype(int)
            
            A, xtrue = gauss1D(iscomplex, n, m)
            Astar = np.conj(A.T)
            xhat = init_angle(xtrue, theta)
            B = np.diag(abs(A@xtrue))   
            D = Astar@np.linalg.inv(B)
            _, dual_sol = basis_pursuit(m,D,xhat)
            alpha = ((dual_sol.T)@xtrue)/((dual_sol.T)@dual_sol)
            dual_sol = alpha * dual_sol
            #sol, res, _ = PhaseMax(n, A, xhat, xtrue)

            res = ((np.linalg.norm(xtrue-dual_sol)))/(np.linalg.norm(xtrue))
            results[p,q] = res
    
    final_results = np.mean(results, axis=1)
    #print("error",final_results)
    plt.plot(ratio,final_results)
    # plt.plot(abs(xtrue)-abs(sol),'g')
    # plt.plot(abs(sol),'r')
    # plt.plot(abs(xtrue),'c')
    plt.show()

gauss_phaseplot(True, num_trials = 10, n = 100, theta = np.pi/8)
# n = 100
# m = 800    
# A, xtrue = gauss1D(True, n, m)
# Astar = np.conj(A.T)
# xhat = init_angle(xtrue, theta = np.pi/8)  
# B = np.diag(abs(A@xtrue))   
# D = Astar@np.linalg.inv(B)
# sol, dual_sol = basis_pursuit(m,D,xhat)
# alpha = ((dual_sol.T)@xtrue)/((dual_sol.T)@dual_sol)
# print(alpha)
# dual_sol = alpha * dual_sol
# plt.plot(abs(dual_sol),'r')
# plt.plot(abs(xtrue),'g')



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


# %%
