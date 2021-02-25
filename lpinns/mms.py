import dolfin as df
import sympy as sp
from dolfin import pi


def poisson(alpha_value=1):
    '''
    We will conder Poisson on [-1, 1]^2 with bcs grad(u).n = 0 everywhere
    but on the top, where we will vary boundary conditions.
    '''
    x, y, alpha = sp.symbols('x[0] x[1] alpha')
    # True also dirichlet data
    u_sp = sp.cos(pi*x)*sp.cos(pi*y) + 1 - (y+1)**2*(x+1)**2*(1-x)**2
    mean = sp.integrate(sp.integrate(u_sp, (x, -1, 1)), (y, -1, 1))

    u_sp = u_sp - mean/4
    grad_u = (u_sp.diff(x, 1), u_sp.diff(y, 1))
    grad_un = [sign*grad_u[i] for sign, i in ((-1, 0), (1, 0), (-1, 1), (1, 1))]
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
            'lm_true': tuple(map(as_expr, grad_un)),
            'g_neumann': as_expr(g_neumann),
            'g_robin': as_expr(g_robin),
            'g_dirichlet': as_expr(u_sp),
            'f': as_expr(f_sp),
            'mean': mean}

# Let's have 1d - maybe the condition number is smaller
def poisson_1d(alpha_value=1, zero_mean=True):
    '''
    We will conder Poisson on [-1, 1]. Bcs speced on left/right
    '''
    x, alpha = sp.symbols('x[0] alpha')
    # True also dirichlet data
    u_sp = sp.cos(pi*x) - x**2
    mean = sp.integrate(u_sp, (x, -1, 1))

    if zero_mean:
        u_sp = u_sp - mean/2
    # Neumann data (left, right)
    g_neumann = (u_sp.diff(x, 1), -u_sp.diff(x, 1)) 
    # For robin -du/dn = alpha*u + g
    g_robin = (u_sp.diff(x, 1) - alpha*u_sp,
               -u_sp.diff(x, 1)- alpha*u_sp)
    # Source term
    f_sp = -u_sp.diff(x, 2)

    as_expr = lambda f, alpha=alpha_value: df.Expression(sp.printing.ccode(f),
                                                         degree=4,
                                                         alpha=alpha)
    # Wrap as expression for FEniCS
    return {'u_true': as_expr(u_sp),
            'g_neumann': tuple(map(as_expr, g_neumann)),
            'g_robin': tuple(map(as_expr, g_robin)),
            'g_dirichlet': as_expr(u_sp),
            'f': as_expr(f_sp),
            'mean': mean}
