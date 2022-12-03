import numpy as np
from src.equations import LinearGasDynamics
from src.problem import Problem
from src.callbacks import PlotCallback

if __name__ == "__main__":
    a, rho_0 = 1.0, 1.0
    equation = LinearGasDynamics(a, rho_0)
    t_end = 1.0
    Nx, xmin, xmax = 100, -2.0, 2.0
    CFL = 0.95

    def g(x):
        if x < 0.0:
            return 0.1
        else:
            return 0.3

    def h(x):
        if x < 0.0:
            return -0.4
        else:
            return 0.2

    def u0(x):
        return np.array([g(x), h(x)])

    def sol(x, t):
        return np.array([0.5*(g(x + a*t) + g(x - a*t)) + \
                         0.5*rho_0/a*(h(x - a*t) - h(x + a*t)),
                         0.5*a/rho_0*(g(x - a*t) - g(x + a*t)) + \
                         0.5*(h(x + a*t) + h(x - a*t))])
    bc = "transparent"
    ylim = [[0.0, 0.6], [-0.5, 0.5]]
    callbacks = [PlotCallback(ylim=None, analytic_sol=sol)]
    problem = Problem(Nx, xmin, xmax, t_end, equation=equation,
                      bc=bc, numerical_flux="godunov", CFL=CFL,
                      callbacks=callbacks)
    problem.solve(u0)
