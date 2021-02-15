# Based on residual formulation
#
#   |-div(grad(u)) - f|^2*ds + |u - g|^2*dx
#


from lpinns.mms import poisson_1d as poisson
from lpinns.fenics_legendre import LegendreBasis, legendre

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np


mesh = UnitIntervalMesh(64)
facet_subdomains = MeshFunction('size_t', mesh, 0, 0)
CompiledSubDomain('near(x[0], 0)').mark(facet_subdomains, 1)
CompiledSubDomain('near(x[0], 1)').mark(facet_subdomains, 2)

ds = Measure('ds', domain=mesh, subdomain_data=facet_subdomains)

# MMS data
alpha_value = 1
data = poisson(alpha_value)

f_data, g_data, u_true = (data[key] for key in ('f', 'g_dirichlet', 'u_true'))

errors, conds, degrees = [], [], (3, 4, 5, 6, 7, 8, 9, 10)
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

fig, ax = plt.subplots()

ax.semilogy(degrees, errors, 'bx-')
ax.set_ylabel('|u-uh|_0', color='blue')
ax.set_xlabel('degree')

ax = ax.twinx()
ax.semilogy(degrees, conds, 'ro-')
ax.set_ylabel('condition number', color='red')

plt.show()
