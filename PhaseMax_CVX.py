# Import packages.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Generate a random non-trivial linear program.
m = 15
n = 10
np.random.seed(1)
s0 = np.random.randn(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

# Define and solve the CVXPY problem.
x = cp.Variable(n)
# prob = cp.Problem(cp.Minimize(c.T@x),
#                  [A @ x <= b])
prob = cp.Problem(cp.Maximize(c.T@x),
                 [cp.norm(cp.multiply((A @ x),1/b),"inf") <= 1])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)

plt.plot(-x0, 'r')
plt.plot(x.value, 'b')
plt.show()
# print("A dual solution is")
# print(prob.constraints[0].dual_value)

