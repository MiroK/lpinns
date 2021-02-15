# Based on Lagrangian
#
#   inner(grad(u), grad(u))*dx + inner(u, p)*ds
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

f_data, g_data, u_true = (data[key] for key in ('f', 'g_dirichlet', 'u_true'))

errors, conds, degrees = [], [], (3, 4, 5, 6, 7, 8, 9, 10)
for degree in degrees:
    # Bulk
    basis_V = LegendreBasis(mesh=mesh, degree=degree)
    # Multiplier are two number!
    basis_Q = [Constant(1), Constant(1)]

    V_elm = VectorElement('Real', mesh.ufl_cell(), 0, len(basis_V))
    Q_elm = VectorElement('Real', mesh.ufl_cell(), 0, len(basis_Q))
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
    p0, p1 = split(p)
    q0, q1 = split(q)
    
    a = (inner(grad(u), grad(v))*dx + inner(p0, v)*ds(1) + inner(p1, v)*ds(2) + 
         inner(q0, u)*ds(1) + inner(q1, u)*ds(2))
    L = inner(f_data, v)*dx + inner(g_data, q0)*ds(1) + + inner(g_data, q1)*ds(2)

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
