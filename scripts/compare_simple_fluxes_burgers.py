import numpy as np
from src.equations import Burgers
from src.problem import Problem
from src.callbacks import PlotCallback
from src.util import plot_sols

if __name__ == "__main__":
    equation = Burgers()
    t_end = 5.0
    Nx, xmin, xmax = 100, -2.0, 2.0
    CFL = 0.95

    def g1(x):
        # should return array which has the size as the number of unknowns
        return np.array([np.exp(-10*(x + 1)**2) - np.exp(-10*(x - 1)**2)])
    def g2(x):
        if x < 0:
            return np.array([0.5])
        else:
            return np.array([1.0])
    def g3(x):
        if x < 0:
            return np.array([-1.0])
        else:
            return np.array([1.0])
    def g4(x):
        if x < 0:
            return np.array([-1.0])
        else:
            return np.array([1.5])

    g = g1
    bc = "transparent"
    ylim = [-1.0, 1.0]
    callbacks = [PlotCallback(ylim=ylim)]
    #callbacks = []
    problems = {}
    for num_flux in ['rusanov', 'LxW', 'godunov', 'roe']:
        problem = Problem(Nx, xmin, xmax, t_end, equation=equation,
                          bc=bc, numerical_flux=num_flux, CFL=CFL,
                          callbacks=callbacks)
        problems[num_flux] = problem
    plot_sols(problems, g,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                g.__name__,
                                                                t_end),
              ylim=ylim, save=False)
