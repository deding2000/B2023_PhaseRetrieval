#%%
import scipy.io
from Helper_functions import *
import time
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
#%%
b = scipy.io.loadmat('Data/Proteindata/b_short.mat')
b = b['bs']
x = scipy.io.loadmat('Data/Proteindata/caffeine_true.mat')
x = x['x'].ravel()
n = len(x)
idx_b = scipy.io.loadmat('Data/Proteindata/idx_bs.mat')
idx_b = idx_b['idx_bs']
filters = scipy.io.loadmat('Data/Proteindata/filter.mat')
filters = filters['filters']
#%%
FFT1 = np.fft.fft(np.eye(n))
FFT2 = FFT1 * np.diag(filters[0][1].ravel())
A = np.concatenate((FFT1,FFT2))
#%%
idx_s = np.squeeze(idx_b)
#%% choose a vectors
A_s = A[:,idx_s.T]
#%%
#print(data['filters'][0][1].shape)
#print(data.keys())
#print(data['N'])
#A = FFT[:,np.random.choice(FFT.shape[0], n, replace=False)]
#b = np.abs(A@x)
#%%
#xhat = spectral_initializer(A, b,n,m)
xhat = init_angle(x, np.pi/180*10)
#%%

#maybe a bit faster with BP
# B = np.diag((b))
# B1 = np.diag(1/(b))
# D = np.conj((A).T)@B1
start_time = time.time()
sol = PhaseMax(A,b,xhat,isComplex=False, verbose = True)
print(time.time() - start_time)
#%%
sol.reshape(n,n)
plt.imshow(sol)
plt.show()

