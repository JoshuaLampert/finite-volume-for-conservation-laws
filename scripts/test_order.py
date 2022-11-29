import numpy as np
from src.equations import Linear
from src.problem import Problem
from src.util import plot_order

if __name__ == "__main__":
    a = 1.0
    equation = Linear(a)
    t_end = 1.0
    Nx, xmin, xmax = 100, 0.0, 1.0
    CFL = 0.95
    bc = "periodic"
    N_gl = 8

    def g(x):
        return np.array([np.sin(2*np.pi*x)**2])

    def analytic_sol(x, t):
        if bc == "transparent":
            return g(x - a*t)
        elif bc == "periodic":
            return g((x - a*t - xmin) % (xmax - xmin) + xmin)
        else:
            raise NotImplementedError()
    callbacks = []
    problems = {}
    for num_flux in ['rusanov', 'LxW', 'godunov', 'roe']:
        problem = Problem(Nx, xmin, xmax, t_end, equation=equation,
                          bc=bc, numerical_flux=num_flux, CFL=CFL,
                          callbacks=callbacks)
        problems[num_flux] = problem 
    for N in [3]:
        problem_ader = Problem(Nx, xmin, xmax, t_end, equation=equation,
                               bc=bc, numerical_flux='ader', N=N, CFL=CFL,
                               Nt_max=int(1e9), N_gl=N_gl, callbacks=callbacks)
        problems["ADER" + str(N)] = problem_ader
    Nxs = 16*2**np.arange(4)
    plot_order(problems, g, analytic_sol, Nxs=Nxs)
