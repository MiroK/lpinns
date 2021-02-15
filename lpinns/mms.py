import dolfin as df
import sympy as sp
from dolfin import pi


def poisson(alpha_value=1):
    '''
    We will conder Poisson on [0, 1]^2 with bcs grad(u).n = 0 everywhere
    but on the top, where we will vary boundary conditions.
    '''
    x, y, alpha = sp.symbols('x[0] x[1] alpha')
    # True also dirichlet data
    u_sp = sp.cos(pi*x)*sp.cos(pi*y) + 1 - y**2*x**2*(1-x)**2
    mean = sp.integrate(sp.integrate(1 - y**2*x**2*(1-x)**2, (x, 0, 1)), (y, 0, 1))

    u_sp = u_sp - mean
    # Neumann data on top edge we give -grad(u).n there
    g_neumann = -u_sp.diff(y, 1)
    # For robin -du/dn = alpha*u + g
    g_robin = -u_sp.diff(y, 1) - alpha*u_sp
    # Source term
    f_sp = -u_sp.diff(x, 2) - u_sp.diff(y, 2)

    as_expr = lambda f, alpha=alpha_value: df.Expression(sp.printing.ccode(f),
                                                         degree=4,
                                                         alpha=alpha)
    # Wrap as expression for FEniCS
    return {'u_true': as_expr(u_sp),
            'g_neumann': as_expr(g_neumann),
            'g_robin': as_expr(g_robin),
            'g_dirichlet': as_expr(u_sp),
            'f': as_expr(f_sp),
            'mean': mean}
