
import numpy as np
from Helper_functions import *
from numpy import linalg as la
import matplotlib.pyplot as plt
import cvxpy as cp

class image_recon():
    def __init__(self, im, num_masks): 
        if len(im.shape) == 2:
            self.n = im.shape[0]*im.shape[1]
        else:
            self.n = im.shape[0]
        self.im = im
        self.x0 = im.ravel()
        self.num_masks = num_masks
    
    def fourier_matrix(self):
        m = self.n*self.num_masks
        DFT = np.fft.fft(np.eye(m.astype(int)))
        
        A = DFT[:,np.random.choice(DFT.shape[0], self.n, replace=False)]
        return A

    def phasemax(self, angle):
        A = self.fourier_matrix()
        b = abs(A@self.x0)
        xhat = init_angle(self.x0, angle)
        x = cp.Variable(self.n, complex=True)
        prob = cp.Problem(cp.Maximize(cp.real(inp(x,xhat))),[cp.norm(cp.multiply(A @ x,1/b), "inf") <= 1])
        prob.solve(solver="ECOS")

        return x.value

    def error(self,xsol):
        alpha = inp(self.x0,xsol)/(inp(xsol,xsol))
        sol = alpha * xsol
        error = la.norm(self.x0-sol)**2/la.norm(self.x0)**2
        return error
    
n = 100
m = np.array([280,290,300,310,320,350,400])
success = 1e-5
repeats = 10
nbsucceded = np.zeros(len(m))
#x0 = np.random.randint(2,size = (n,n))
for i, m1 in enumerate(m):
        for j in range(repeats):
            x0 = np.random.random(n) + 1j*np.random.random(n)
            im_class = image_recon(x0,m1/100)
            xval = im_class.phasemax(angle = np.pi*180/25)
            error = im_class.error(xval)
            if error < success:
                    nbsucceded[i] += 1
print( nbsucceded/repeats*100)



    # def fouriertran(self, x):
    #     is_cp = isinstance(x, cp.Variable)
    #     if is_cp:
    #         x = x.value
    #     random_vars = np.random.rand(self.num_masks,self.n,self.n) 
    #     mask3d = (random_vars<0.5)*2-1
    #     measurements = np.zeros((self.num_masks,self.n,self.n))
    #     for i in range(self.num_masks):
    #         this_mask = mask3d[i,:,:]
    #         measurements[i,:,:] = np.fft.fft2(np.multiply(x,this_mask))
    #     if is_cp:
    #         measurements_ret = cp.Variable(len(measurements.ravel()), value=measurements.ravel())
    #     else:
    #         measurements_ret = measurements.ravel()
    #     return measurements_ret