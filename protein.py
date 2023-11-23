import scipy.io
from Helper_functions import *
def spectral_initializer(A,b,n,m, truncate = True, isScaled = False, optimal = False):
    mu = np.mean(b**2)
    if truncate:
        mu = np.mean(b**2)
        b0 = b**2 <= 3**2 * mu
        # for i in range(m):
        #     if b[i]**2 > 3**2 * mu:
        #         b0[i] = 0
        print(m - np.sum(b0))
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

data = scipy.io.loadmat('bs.mat')
b = data['bs']
m = len(b)
x = data['x'].ravel()

#print(data['filters'][0][1].shape)
#print(data.keys())
#print(data['N'])
n = len(x)
m = n
FFT = np.fft.fft(np.eye(m))
A = FFT * np.diag(data['filters'][0][1].ravel())
#A = FFT[:,np.random.choice(FFT.shape[0], n, replace=False)]
b = np.abs(A@x)
#xhat = spectral_initializer(A, b,n,m)
xhat = np.random.normal(0,1,n)
sol = PhaseMax(A,b,xhat,isComplex=False, verbose = False)
sol.reshape(n,n)
plt.imshow(sol)
plt.show()

