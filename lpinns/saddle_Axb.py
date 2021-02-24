import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from scipy.sparse import diags, bmat
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

n = 10
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, 'RT', 1)
Q = FunctionSpace(mesh, 'DG', 0)
W = [V, Q]
sigma, u = map(TrialFunction, W)
tau, v = map(TestFunction, W)

a00 = inner(sigma, tau)*dx
a01 = inner(u, div(tau))*dx
a10 = inner(v, div(sigma))*dx

A, Bt, B = map(lambda a: assemble(a).array(), (a00, a01, a10))
AA = bmat([[A, Bt],
           [B, None]]).todense()

b = np.random.rand(V.dim()+Q.dim())

x_true = np.linalg.solve(AA, b)

b0, b1 = b[:V.dim()], b[V.dim():]

# We are minimizing 0.5*(x, A*x) - (b0, x)
# subject to Bx - b1 = 0

A, B, b0, b1, x_true = map(lambda x: torch.tensor(x, dtype=torch.float64),
                           (A, B, b0, b1, x_true))

x = nn.Parameter(torch.rand(V.dim(), dtype=torch.float64), requires_grad=True)
y = nn.Parameter(torch.zeros(Q.dim(), dtype=torch.float64), requires_grad=True)

nsteps = 1000

# loss = (0.5*torch.sum(x*(torch.matmul(A, x) - 2*b0)) +
#         torch.sum(y*(torch.matmul(B, x) - b1)))
# loss.backward()
# torch.matmul(A, x) + torch.matmul(y, B) - b0  is x.grad
# torch.matmul(B, x) - b1 is y.grad


optimizer = optim.LBFGS([x], max_iter=1000,
                        history_size=1000, tolerance_change=1e-12,
                        line_search_fn="strong_wolfe")
    
def closure():
    optimizer.zero_grad()
        
    loss = (0.5*torch.sum(x*(torch.matmul(A, x) - 2*b0)) +
            torch.sum(y*(torch.matmul(B, x) - b1)))
    loss.backward()
    error = torch.norm(x_true - torch.cat([x, y]))
    print(f'\t loss = {float(loss)}, {error}')
    
    return loss

lr = 1E-1
for step in range(nsteps):
    print(f'Step {step}')
    # Fix y and minimize over x
    y.grad = None
    optimizer.step(closure)

    x.grad = None
    y.grad = None
    loss = (0.5*torch.sum(x*(torch.matmul(A, x) - 2*b0)) +
            torch.sum(y*(torch.matmul(B, x) - b1)))
    loss.backward()

    # Ascent with what is gradient of loss wrt y @ new x
    with torch.no_grad():
        y += lr*y.grad


x_true = x_true.detach().numpy().flatten()
x_true, y_true = x_true[:V.dim()], x_true[V.dim():]
        
fig, ax = plt.subplots(1, 2)
ax = ax.ravel()

ax[0].plot(x_true, 'rx', label='true')
ax[0].plot(x.detach().numpy().flatten())

ax[1].plot(y_true, 'rx', label='true')
ax[1].plot(y.detach().numpy().flatten())

plt.show()
