import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.mesh import Mesh
    from src.callbacks import StepsizeCallback
    from src.equations import LinearScalar
    from src.problem import Problem
    from src.util import plot_order

    a = 1.0
    equation = LinearScalar(a)
    t_end = 1.0
    xmin, xmax, Nx = 0.0, 1.0, 100
    mesh = Mesh(xmin, xmax, 0.0, t_end, Nx, dt=1.0)
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
    callbacks = [StepsizeCallback(equation, mesh, CFL=CFL)]
    problems = {}
    for num_flux in ['rusanov', 'godunov', 'roe']:
        problem = Problem(mesh, equation=equation,
                          bc=bc, numerical_flux=num_flux,
                          callbacks=callbacks)
        problems[num_flux] = problem
    for N in range(1, 7):
        problem_ader = Problem(mesh, equation=equation,
                               bc=bc, numerical_flux='ader', N=N,
                               Nt_max=int(1e9), N_gl=N_gl, callbacks=callbacks)
        problems["ADER" + str(N)] = problem_ader
    Nxs = 16*2**np.arange(4)
    plot_order(problems, g, analytic_sol, Nxs=Nxs, save=False,
               error_type=np.inf)
