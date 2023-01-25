import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.mesh import Mesh
    from src.callbacks import StepsizeCallback, PlotCallback
    from src.equations import ShallowWater
    from src.problem import Problem
    from src.util import plot_sols

    g = 1.0
    equation = ShallowWater(g)
    xmin, xmax, Nx = -1.0, 1.0, 100
    t_end = 1.0
    mesh = Mesh(xmin, xmax, 0.0, t_end, Nx, dt=1.0)
    CFL = 0.95

    def u0(x):
        if x < 0.0:
            return np.array([1.1, 0.0])
        else:
            return np.array([0.9, 0.0])
    ylim = [[0.8, 1.2], [-0.1, 0.2]]
    bc = "transparent"
    callbacks = [StepsizeCallback(equation, mesh, CFL=CFL),
                 PlotCallback(equation, ylim=ylim, prim=False)]
    # callbacks = [StepsizeCallback(equation, mesh, CFL=CFL)]
    problems = {}
    for num_flux in ["eigen", "hll", "rusanov"]:
        problem = Problem(mesh, equation=equation, bc=bc,
                          numerical_flux=num_flux, callbacks=callbacks)
        problems[num_flux] = problem
    plot_sols(problems, u0,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                u0.__name__,
                                                                t_end),
              ylim=ylim, save=False)
