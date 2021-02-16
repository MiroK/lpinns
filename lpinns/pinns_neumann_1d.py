# Based on residual formulation
#
#   |-div(grad(u)) - f|^2*ds + |-du/dn - g|^2*dx
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
data = poisson(alpha_value)
# NOTE: No cvrg

f_data, g_data, u_true = (data[key] for key in ('f', 'g_neumann', 'u_true'))

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
    n = FacetNormal(mesh)
    F = (inner(-div(grad(u))-f_data, -div(grad(v)))*dx +
         inner(-dot(grad(u), n) + g_data[0], -dot(grad(v), n))*ds(1) +
         inner(-dot(grad(u), n) + g_data[1], -dot(grad(v), n))*ds(2))
#         inner(-dot(grad(u), n) + g_data[0], v)*ds(1) +
#         inner(-dot(grad(u), n) + g_data[1], v)*ds(2))
         
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

# FIXME: plot solution
#        where is the error

V = FunctionSpace(mesh, 'CG', 1)
x = V.tabulate_dof_coordinates().reshape((-1, ))
idx = np.argsort(x)

uh = project(uh, V).vector().get_local()[idx]
u_true = interpolate(u_true, V).vector().get_local()[idx]

plt.figure()
plt.plot(x, uh, label='uh')
plt.plot(x, u_true, label='u')
plt.plot(x, uh - u_true, label='uh - u')
plt.legend()
plt.show()
