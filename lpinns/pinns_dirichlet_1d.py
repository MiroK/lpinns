# Based on residual formulation
#
#   |-div(grad(u)) - f|^2*ds + |u - g|^2*dx
#


from lpinns.mms import poisson_1d as poisson
from lpinns.fenics_legendre import LegendreBasis, legendre

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np


mesh = IntervalMesh(64, -1, 1)
facet_subdomains = MeshFunction('size_t', mesh, 0, 0)
CompiledSubDomain('near(x[0], -1)').mark(facet_subdomains, 1)
CompiledSubDomain('near(x[0], 1)').mark(facet_subdomains, 2)

ds = Measure('ds', domain=mesh, subdomain_data=facet_subdomains)

# MMS data
alpha_value = 1
zero_mean = False
data = poisson(alpha_value, zero_mean=zero_mean)#=False)
# NOTE: zero_mean == True -> converges fine
#       zero_mean == False -> no way

f_data, g_data, u_true = (data[key] for key in ('f', 'g_dirichlet', 'u_true'))

errors, conds, degrees = [], [], list(range(3, 12))
for degree in degrees:
    # Bulk
    basis_V = LegendreBasis(mesh=mesh, degree=degree, min_degree=2)
    V = VectorFunctionSpace(mesh, 'Real', 0, len(basis_V))
    # These are just coefficients to be computed what weight the
    # linear combinations of basis functions
    u, v = TrialFunction(V), TestFunction(V)
    # Combine with basis functions
    u = sum(u[i]*fi for i, fi in enumerate(basis_V))
    v = sum(v[i]*fi for i, fi in enumerate(basis_V))
    #
    # Energy gorm derif foo(u)^2 -> foo(u)*[d foo / d u][v]
    #
    F = inner(-div(grad(u))-f_data, -div(grad(v)))*dx + inner(u - g_data, v)*ds
    a, L = lhs(F), rhs(F)
    
    uh = Function(V)
    A, b = map(assemble, (a, L))
    # Some conditioning info
    lmin, lmax = np.sort(np.abs(np.linalg.eigvalsh(A.array())))[[0, -1]]
    cond = lmax/lmin
    print('lmin = {:.4E}, lmax = {:.4E}, cond = {:.4E}'.format(lmin, lmax, cond))
    solve(A, uh.vector(), b)
    # Recombine again
    uh = sum(uh[i]*fi for i, fi in enumerate(basis_V))
    
    error = sqrt(abs(assemble(inner(uh - u_true, uh - u_true)*dx)))
    errors.append(error)
    conds.append(cond)
    print(degree, error)

fig, ax = plt.subplots(figsize=(16, 10))

ax.semilogy(degrees, errors, 'bx-')
ax.set_ylabel('|u-uh|_0', color='blue')
ax.set_xlabel('degree')

ax = ax.twinx()
ax.semilogy(degrees, conds, 'ro-')
ax.set_ylabel('condition number', color='red')
fig.savefig('pinns_D_zeroMean%d_cvrg.pdf' % zero_mean)
# Solution

V = FunctionSpace(mesh, 'CG', 1)
x = V.tabulate_dof_coordinates().reshape((-1, ))
idx = np.argsort(x)

uh = project(uh, V).vector().get_local()[idx]
u_true = interpolate(u_true, V).vector().get_local()[idx]

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(x, uh, label='uh')
ax.plot(x, u_true, label='u')
ax.plot(x, uh - u_true, label='uh - u')
plt.legend()

fig.savefig('pinns_D_zeroMean%d_error.pdf' % zero_mean)
plt.show()
