# %%
import numpy as np
from numpy import linalg as la
from Helper_functions import *

n = 10
m = 50
Data = GaussData(n,m,True)
xhat = init_angle(Data.x0, np.partitioni/2)
xsol = PhaseMax(Data.A, Data.b, xhat,verbose=True)

# %%