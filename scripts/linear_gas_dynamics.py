import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.mesh import Mesh
    from src.callbacks import StepsizeCallback, PlotCallback
    from src.equations import LinearGasDynamics
    from src.problem import Problem
    from src.util import plot_sols

    a, rho_0 = 1.0, 1.0
    equation = LinearGasDynamics(a, rho_0)
    t_end = 1.0
    xmin, xmax, Nx = -2.0, 2.0, 100
    mesh = Mesh(xmin, xmax, 0.0, t_end, Nx, dt=1.0)
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
        return np.array([0.5*(g(x + a*t) + g(x - a*t)) +
                         0.5*rho_0/a*(h(x - a*t) - h(x + a*t)),
                         0.5*a/rho_0*(g(x - a*t) - g(x + a*t)) +
                         0.5*(h(x + a*t) + h(x - a*t))])
    bc = "transparent"
    ylim = [[-0.2, 0.6], [-0.5, 0.5]]
    callbacks = [StepsizeCallback(equation, mesh, CFL=CFL),
                 PlotCallback(equation, ylim=ylim, analytic_sol=sol)]
    # callbacks = [StepsizeCallback(equation, mesh, CFL=CFL)]
    problems = {}
    for num_flux in ["rusanov", "godunov"]:
        problem = Problem(mesh, equation=equation, bc=bc,
                          numerical_flux=num_flux, callbacks=callbacks)
        problems[num_flux] = problem

    def ana_sol(x): return sol(x, t_end)
    plot_sols(problems, u0,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                u0.__name__,
                                                                t_end),
              ylim=ylim, save=False, analytic_sol=ana_sol)
