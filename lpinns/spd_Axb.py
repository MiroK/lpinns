import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from scipy.sparse import diags
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

n = 100
mesh = IntervalMesh(n, 0, 1)
V = FunctionSpace(mesh, 'CG', 1)
u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = inner(Constant(0), v)*dx
bcs = DirichletBC(V, Constant(0), 'on_boundary')
A, _ = assemble_system(a, L, bcs)
A = A.array()

b = np.random.rand(len(A))
b[list(bcs.get_boundary_values().keys())] = 0.

x_true = np.linalg.solve(A, b)
print('symmetry', np.linalg.norm(A.T - A), np.linalg.cond(A))

x_true = torch.tensor(x_true, dtype=torch.float64)

A = torch.tensor(A, dtype=torch.float64)
b = torch.tensor(b, dtype=torch.float64)
x = nn.Parameter(torch.rand(b.shape, dtype=torch.float64), requires_grad=True)

params = [x]
optimizer = optim.Adam(params, lr=1E0)

nsteps = 10000
for step in range(nsteps):
    optimizer.zero_grad()

    loss = 0.5*torch.sum(x*(torch.matmul(A, x) - 2*b))
    loss.backward()

    with torch.no_grad():
        error = torch.norm(x_true - x)    
    print(f'loss = {float(loss)}, {error}')
    optimizer.step()
    

if False:
    optimizer = optim.LBFGS(params, max_iter=10000,
                            history_size=1000, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    
    def closure():
        optimizer.zero_grad()
        
        loss = 0.5*torch.sum(x*(torch.matmul(A, x) - 2*b))
        loss.backward()
        error = torch.norm(x_true - x)
        print(f'loss = {float(loss)}, {error}')
        
        return loss


    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch = {epoch}")
        optimizer.step(closure)

plt.figure()
plt.plot(x_true.detach().numpy().flatten(), 'rx', label='true')
plt.plot(x.detach().numpy().flatten())
plt.legend()
plt.show()
