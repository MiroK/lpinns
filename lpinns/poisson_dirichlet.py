# Based on Lagrangian
#
#   inner(grad(u), grad(u))*dx + inner(u, p)*ds1
#


from lpinns.mms import poisson
from lpinns.fenics_legendre import LegendreBasis, legendre

import matplotlib.pyplot as plt
from dolfin import *
import numpy as np


mesh = UnitSquareMesh(64, 64)
facet_subdomains = MeshFunction('size_t', mesh, 1, 0)
CompiledSubDomain('near(x[1], 1)').mark(facet_subdomains, 1)

ds = Measure('ds', domain=mesh, subdomain_data=facet_subdomains)

# MMS data
alpha_value = 1
data = poisson(alpha_value)

f_data, g_data, u_true = (data[key] for key in ('f', 'g_dirichlet', 'u_true'))

errors, conds, degrees = [], [], (3, 4, 5, 6, 7)
for degree in degrees:
    alpha = Constant(alpha_value)
    # Bulk
    basis_V = LegendreBasis(mesh=mesh, degree=degree)
    # Multiplier on {y = 1} so we only vary in x
    x = SpatialCoordinate(mesh)[0]
    basis_Q = [legendre(deg, x) for deg in range(degree-1)]

    V_elm = VectorElement('Real', triangle, 0, len(basis_V))
    Q_elm = VectorElement('Real', triangle, 0, len(basis_Q))
    W_elm = MixedElement([V_elm, Q_elm])
    # These are just coefficients to be computed what weight the
    # linear combinations of basis functions
    W = FunctionSpace(mesh, W_elm)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    # Combine with basis functions
    u = sum(u[i]*fi for i, fi in enumerate(basis_V))
    v = sum(v[i]*fi for i, fi in enumerate(basis_V))
    # Combine with basis functions
    p = sum(p[i]*fi for i, fi in enumerate(basis_Q))
    q = sum(q[i]*fi for i, fi in enumerate(basis_Q))
    
                          
    a = (inner(grad(u), grad(v))*dx + inner(p, v)*ds(1) +
         inner(q, u)*ds(1))
    L = inner(f_data, v)*dx + inner(g_data, q)*ds(1)

    wh = Function(W)
    A, b = map(assemble, (a, L))
    # Some conditioning info
    lmin, lmax = np.sort(np.abs(np.linalg.eigvalsh(A.array())))[[0, -1]]
    cond = lmax/lmin
    print('lmin = {:.4E}, lmax = {:.4E}, cond = {:.4E}'.format(lmin, lmax, cond))
    solve(A, wh.vector(), b)
    # Recombine again
    uh, ph = wh.split(deepcopy=True)
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
