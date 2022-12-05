import numpy as np

from src.callbacks import PlotCallback
from src.equations import ShallowWater
from src.problem import Problem
from src.util import plot_sols

if __name__ == "__main__":
    g = 1.0
    equation = ShallowWater(g)
    Nx, xmin, xmax = 100, -1.0, 1.0
    t_end = 1.0
    CFL = 0.95

    def u0(x):
        if x < 0.0:
            return np.array([1.1, 0.0])
        else:
            return np.array([0.9, 0.0])
    ylim = [[0.8, 1.2], [-0.1, 0.2]]
    bc = "transparent"
    callbacks = [PlotCallback(ylim=ylim, equation=equation, prim=False)]
    # callbacks = []
    problems = {}
    for num_flux in ["rusanov", "LxW"]:
        problem = Problem(Nx, xmin, xmax, t_end, equation=equation,
                          bc=bc, numerical_flux=num_flux, CFL=CFL,
                          callbacks=callbacks)
        problems[num_flux] = problem
    plot_sols(problems, u0,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                u0.__name__,
                                                                t_end),
              ylim=ylim, save=False)
