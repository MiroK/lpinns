import sympy as sp
import dolfin as df
import itertools
from functools import reduce
import operator
import ufl


def sympy_to_ufl(expr, substitutions):
    '''Mimimal translator that handles operators in Legendre polyn. expressions'''
    # Terminals
    if not expr.args:
        if expr in substitutions:
            return substitutions[expr]

        if expr.is_number:
            return df.Constant(float(expr))

        raise ValueError(str(expr))

    rule = {sp.Add: ufl.algebra.Sum,
            sp.Mul: ufl.algebra.Product,
            sp.Pow: ufl.algebra.Power} 
    # Compounds. NOTE that sympy operands can have many args wherase
    # the above ufl guys have only two at most
    args = tuple(sympy_to_ufl(arg, substitutions) for arg in expr.args)
    if len(args) < 3:
        return rule[type(expr)](*args)
    # So we get the first arg and ask for the other
    # (+ 3 4 5) -> (+ 3 (+ 3 4))
    head, tail = args[0], args[1:]
    return rule[type(expr)](head, sympy_to_ufl(type(expr)(*expr.args[1:]), substitutions))


def legendre(degree, var):
    '''UFL Legendre 1d polynomial of given degree'''
    var_ = sp.Symbol('x')
    l = sp.legendre(degree, var_)
    # Translated from one symbolic representation to another
    return sympy_to_ufl(l, {var_: var})


def LegendreBasis(mesh, degree, min_degree=0):
    '''Tensor product basis of Legendre polynomials'''
    coords = df.SpatialCoordinate(mesh)
    # We want to build a tensor product space. We will generate symbolic
    # code in ufl because with expressions derivatives might get tedious
    basis = [[legendre(deg, xi) for deg in range(min_degree, degree+1)]
             for xi in coords]
    # The tensor product 
    basis = [reduce(operator.mul, b) for b in itertools.product(*basis)]
    return basis

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dolfin import *
    import numpy as np
    
    mesh = IntervalMesh(100, -1, 1)
    V = FunctionSpace(mesh, 'CG', 1)
    foos = LegendreBasis(mesh=mesh, degree=3)
    # Check out a few of them
    plt.figure()
    for f in foos:
        fh = project(f, V)
        x = V.tabulate_dof_coordinates().reshape((-1, ))
        y = fh.vector().get_local()
        idx = np.argsort(x)
        l, = plt.plot(x[idx], y[idx])
    plt.show()
