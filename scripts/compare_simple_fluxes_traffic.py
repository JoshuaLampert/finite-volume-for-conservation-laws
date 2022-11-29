import numpy as np
from src.equations import Traffic
from src.problem import Problem
from src.callbacks import PlotCallback
from src.util import plot_sols

if __name__ == "__main__":
    equation = Traffic()
    t_end = 1.0
    Nx, xmin, xmax = 100, -2.0, 2.0
    CFL = 0.95
    def g(x):
        return np.array([0.2*np.exp(-10*(x + 1)**2) - \
                         0.2*np.exp(-10*(x - 1)**2) + 0.4])
    bc = "periodic"
    ylim = [0.1, 0.8]
    #callbacks = [PlotCallback(ylim=ylim)]
    callbacks = []
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
