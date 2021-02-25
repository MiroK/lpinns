# Work from Lagrangian

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from shapePINNs.calculus import div, grad, transpose, dot
import shapePINNs.reference_domain as domains

from itertools import chain
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

    bcs = df.DirichletBC(V, g_, 'on_boundary')
    
    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = df.inner(f_, v)*df.dx

    uh = df.Function(V)
    df.solve(a == L, uh, bcs)

    return uh


def viz(domain, u, lm, fenics):
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
    ax.set_title('FEniCS')
    ax.axis('equal')
    plt.colorbar(c)

    fig, ax = plt.subplots()
    c = ax.tricontourf(triang, num)
    ax.set_title('NN U')
    ax.axis('equal')    
    plt.colorbar(c)

    fig, ax = plt.subplots()
    c = ax.tricontourf(triang, np.abs(true-num), levels=10)
    ax.set_title('|error|')
    ax.axis('equal')    
    plt.colorbar(c)

    num = lm(grid).squeeze(0).detach().numpy().flatten()
    fig, ax = plt.subplots()
    c = ax.tricontourf(triang, num)
    ax.set_title('NN LM')
    ax.axis('equal')    
    plt.colorbar(c)

    grad_uh = df.project(-df.grad(fenics), df.VectorFunctionSpace(domain.mesh(), 'DG', 0))

    fig, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    for tag in (1, ): #2, 3, 4):
        ax = axs[tag-1]
        
        area_x, area_q, area_n = domain.get_facet_quadrature_mesh(2, tag)
        g = torch.tensor([np.array([grad_uh(p) for p in area_x.squeeze(0)])])
        fen = dot(g, area_n).squeeze(0).detach().numpy()
        l, = ax.plot(fen, marker='x', linestyle='none')
        
        num = lm(area_x).squeeze(0).detach().numpy()
        ax.plot(num, color=l.get_color())
        ax.axis('equal')
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

    
class LagrangeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.lin1 = nn.Linear(2, 32)
        self.lin2 = nn.Linear(32, 32)        
        self.lin3 = nn.Linear(32, 1)
 
    def forward(self, x):
        # A network for displacement
        y = self.lin1(x)
        y = torch.tanh(y)
        y = self.lin2(y)
        y = torch.tanh(y)
        y = self.lin3(y)

        y = y.squeeze(2)
        
        return y


u = PoissonNetwork()
u.double()

lm = LagrangeNetwork()
lm.double()

domain = domains.EllipseDomain2D([0, 0], 2, 1, 0.125)
# Use quadrature points to enforce residuals
volume_x, volume_q = domain.get_volume_quadrature_mesh(1)    
volume_x.requires_grad = True

# Use quadrature points to enforce residuals
area_x, area_q, area_n = domain.get_facet_quadrature_mesh(20)
area_x.requires_grad = True

f_data = lambda x: torch.ones_like(x[..., 0])  # np.pi**2*torch.sin(np.pi*x[..., 0])
g_data = lambda x: torch.ones_like(x[..., 0])  # torch.sin(np.pi*x[..., 0])


optimizer_u = optim.LBFGS(u.parameters(), max_iter=50,
                          history_size=1000, tolerance_change=1e-10,
                          line_search_fn="strong_wolfe")

# FEniCS reference
uh = fenics_solution(mesh=domain.mesh(), f=f_data, g=g_data)
uh_values = torch.tensor(np.array([uh(xi) for xi in volume_x.detach().numpy().squeeze(0)]),
                         dtype=torch.float64)

def sleep_handler(signum, frame, domain=domain, u=u, lm=lm, true=uh):
    viz(domain, u, lm, true)

epsilon = 1E4
loss_history = []
def closure(loss_history=loss_history, start='robin', add_lagrange=False):
    optimizer_u.zero_grad()

    grad_u = grad(u(volume_x), volume_x)

    if start == 'robin':
        # Volume part
        loss = (
            0.5*(dot(grad_u, grad_u)*volume_q).sum()+
            (-f_data(volume_x)*u(volume_x)*volume_q).sum()+
            0.5*(u(area_x)*(epsilon*(u(area_x)-2*g_data(area_x)))*area_q).sum()
        )
    elif start == 'nitsche':
        grad_ua = grad(u(area_x), area_x)
        
        loss = (
            0.5*(dot(grad_u, grad_u)*volume_q).sum()+
            (-f_data(volume_x)*u(volume_x)*volume_q).sum()+
            # Nitsche terms
            -((dot(grad_ua, area_n)*(u(area_x)-g_data(area_x)))*area_q).sum()
            +(epsilon/2*u(area_x)*(u(area_x) - 2*g_data(area_x))*area_q).sum()
        )
    else:
        assert ValueError

    if add_lagrange:
        loss = loss + (lm(area_x)*(u(area_x) - g_data(area_x))*area_q).sum()
        
    loss.backward()

    signal.signal(signal.SIGTSTP, sleep_handler)    

    with torch.no_grad():
        u_nn = u(volume_x)
        error = torch.norm(uh_values - u_nn)
        loss_history.append(float(error))
        
    print(f'\t ({epsilon} {len(loss_history)}) loss = {float(loss)} {float(error)}')    
    
    return loss


viz(domain, u, lm, uh)


nsteps = 10000
lr_lm = 1E-1
for step in range(nsteps):
    loss_history.clear()
    print(f'Step {step}')

    for p in lm.parameters():
        p.grad = None

    optimizer_u.step(closure)

    for p in chain(lm.parameters(), u.parameters()):
        p.grad = None
    
    grad_u = grad(u(volume_x), volume_x)        
    loss = (
        0.5*(dot(grad_u, grad_u)*volume_q).sum()+
        (-f_data(volume_x)*u(volume_x)*volume_q).sum()
        +(lm(area_x)*(u(area_x) - g_data(area_x))*area_q).sum()
    )
    loss.backward()
    print('\t', )
    with torch.no_grad():
        for param in lm.parameters():
            print(torch.norm(param.grad))
            param += lr_lm*param.grad

    signal.signal(signal.SIGTSTP, sleep_handler)

    # epsilon /= 1.0
    # epsilon = min(epsilon, epsilon if step < 20 else 0)
