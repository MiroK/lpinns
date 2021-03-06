# Work from energy formulation of Robin-Poisson

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from shapePINNs.calculus import div, grad, transpose, dot
from shapePINNs.reference_domain import LegGaussDomain2D

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import signal

import dolfin as df


def fenics_solution(mesh, f, g):
    '''-Delta u = f with g on boundary'''
    V = df.FunctionSpace(mesh, 'CG', 1)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    x = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    
    f_ = df.Function(V)
    values = f(torch.tensor([x]))
    f_.vector().set_local(values.detach().numpy().flatten())

    g_ = df.Function(V)
    values = g(torch.tensor([x]))
    g_.vector().set_local(values.detach().numpy().flatten())
    
    a = df.inner(df.grad(u), df.grad(v))*df.dx + df.inner(u, v)*df.ds
    L = df.inner(f_, v)*df.dx

    bcs = df.DirichletBC(V, g_, 'on_boundary')
    
    uh = df.Function(V)
    df.solve(a == L, uh, bcs)

    return uh


def viz(domain, u, fenics):
    '''Compare'''
    grid = 1.0*domain.mesh().coordinates()
    x, y = grid.T
    grid = torch.tensor([grid])
    num = u(grid).squeeze(0).detach().numpy().flatten()
    
    triang = mtri.Triangulation(x, y, domain.mesh().cells())
    # FEniCS is on deformed already
    true = fenics.compute_vertex_values()

    fig, ax = plt.subplots()
    c = ax.tricontourf(triang, true)
    ax.set_title('True')
    plt.colorbar(c)

    fig, ax = plt.subplots()
    c = ax.tricontourf(triang, num)
    ax.set_title('PINNs')
    plt.colorbar(c)

    fig, ax = plt.subplots()
    c = ax.tricontourf(triang, np.abs(true-num), levels=10)
    ax.set_title('|true-PINNs|')
    plt.colorbar(c)

    plt.show()


class PoissonNetwork(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.lin1 = nn.Linear(2, 48)
        self.lin2 = nn.Linear(48, 48)
        self.lin3 = nn.Linear(48, 32)        
        self.lin4 = nn.Linear(32, 1)
 
    def forward(self, x):
        # A network for displacement
        y = self.lin1(x)
        y = torch.tanh(y)
        y = self.lin2(y)
        y = torch.tanh(y)
        y = self.lin3(y)
        y = torch.tanh(y)        
        y = self.lin4(y)
        # FIXME: for my grad the scalar shape is batch x point_index
        # and not batch x point_index x 1
        y = y.squeeze(2)
        
        return y

u = PoissonNetwork()
u.double()


maxiter = 1000
optimizer = optim.LBFGS(u.parameters(), max_iter=maxiter,
                        history_size=1000, tolerance_change=1e-12,
                        line_search_fn="strong_wolfe")

domain = LegGaussDomain2D(32, 32)
# Use quadrature points to enforce residuals
volume_x, volume_q = domain.get_volume_quadrature_mesh(1)    
volume_x.requires_grad = True

# Use quadrature points to enforce residuals
area_x, area_q, area_n = domain.get_facet_quadrature_mesh(20)
area_x.requires_grad = True

f_data = lambda x: torch.ones_like(x[..., 0])  # np.pi**2*torch.sin(np.pi*x[..., 0])
g_data = lambda x: torch.ones_like(x[..., 0])  # torch.sin(np.pi*x[..., 0])

# FEniCS reference
uh = fenics_solution(mesh=domain.mesh(), f=f_data, g=g_data)

epoch_loss = []
gamma = 100
def closure(history=epoch_loss):
    optimizer.zero_grad()

    grad_u = grad(u(volume_x), volume_x)
    grad_ua = grad(u(area_x), area_x)    
    # Volume part
    loss = (
        0.5*(dot(grad_u, grad_u)*volume_q).sum()+
        (-f_data(volume_x)*u(volume_x)*volume_q).sum()+
        # Nitsche terms
        -((dot(grad_ua, area_n)*(u(area_x)-g_data(area_x)))*area_q).sum()
        +(gamma/2*u(area_x)*(u(area_x) - 2*g_data(area_x))*area_q).sum()
    )
    
    print(f'Loss @ {len(history)} = {float(loss)}')
    loss.backward()

    history.append((float(loss), ))
    
    return loss


def interuppt_handler(signum, frame):
    print('Caught CTRL+C!!!')
    raise AssertionError


def sleep_handler(signum, frame, domain=domain, u=u, true=uh):
    viz(domain, u, true)

try:
    epoch_loss.clear()

    signal.signal(signal.SIGINT, interuppt_handler)
    signal.signal(signal.SIGTSTP, sleep_handler)
    
    epochs = 1
    for epoch in range(epochs):
        print(f"Epoch = {epoch}")
        optimizer.step(closure)
        
except AssertionError:
    pass

closure()

viz(domain, u, uh)

print("Done")
