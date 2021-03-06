# Based on energy functional
#
#   inner(grad(u), grad(u))*dx  + inner(u, v)*ds
#


from lpinns.mms import poisson_1d as poisson
from lpinns.fenics_legendre import LegendreBasis

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np


mesh = IntervalMesh(64, -1, 1)
facet_subdomains = MeshFunction('size_t', mesh, 0, 0)
CompiledSubDomain('near(x[0], -1)').mark(facet_subdomains, 1)
CompiledSubDomain('near(x[0], 1)').mark(facet_subdomains, 2)

ds = Measure('ds', domain=mesh, subdomain_data=facet_subdomains)

# MMS data
alpha_value = 1.0
data = poisson(alpha_value)

f_data, g_data, u_true = (data[key] for key in ('f', 'g_robin', 'u_true'))

errors, conds, degrees = [], [], list(range(3, 12))
for degree in degrees:
    alpha = Constant(alpha_value)
    basis = LegendreBasis(mesh=mesh, degree=degree)
    # These are just coefficients to be computed what weight the
    # linear combinations of basis functions
    V = VectorFunctionSpace(mesh, 'R', 0, len(basis))
    u, v = TrialFunction(V), TestFunction(V)
    # Combine with basis functions
    u = sum(u[i]*fi for i, fi in enumerate(basis))
    v = sum(v[i]*fi for i, fi in enumerate(basis))
                          
    a = inner(grad(u), grad(v))*dx + alpha*inner(u, v)*ds(1) + alpha*inner(u, v)*ds(2)
    L = inner(f_data, v)*dx - inner(g_data[0], v)*ds(1) - inner(g_data[1], v)*ds(2)

    uh = Function(V)
    A, b = map(assemble, (a, L))
    # Some conditioning info
    lmin, lmax = np.sort(np.abs(np.linalg.eigvalsh(A.array())))[[0, -1]]
    cond = lmax/lmin
    print('lmin = {:.4E}, lmax = {:.4E}, cond = {:.4E}'.format(lmin, lmax, cond))
    solve(A, uh.vector(), b)
    # Recombine again
    uh = sum(uh[i]*fi for i, fi in enumerate(basis))
    
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
