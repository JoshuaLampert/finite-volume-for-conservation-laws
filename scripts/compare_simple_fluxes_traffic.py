import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.mesh import Mesh
    from src.callbacks import StepsizeCallback, PlotCallback
    from src.equations import Traffic
    from src.problem import Problem
    from src.util import plot_sols

    equation = Traffic()
    t_end = 1.0
    xmin, xmax, Nx = -2.0, 2.0, 100
    mesh = Mesh(xmin, xmax, 0.0, t_end, Nx, dt=1.0)
    CFL = 0.95

    def g(x):
        return np.array([0.2*np.exp(-10*(x + 1)**2) -
                         0.2*np.exp(-10*(x - 1)**2) + 0.4])
    bc = "periodic"
    ylim = [[0.1, 0.8]]
    callbacks = [StepsizeCallback(equation, mesh, CFL=CFL),
                 PlotCallback(equation, ylim=ylim)]
    # callbacks = [StepsizeCallback(equation, mesh, CFL=CFL)]
    problems = {}
    for num_flux in ['eigen', 'hll', 'rusanov', 'godunov', 'roe']:
        problem = Problem(mesh, equation=equation, bc=bc,
                          numerical_flux=num_flux, callbacks=callbacks)
        problems[num_flux] = problem
    plot_sols(problems, g,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                g.__name__,
                                                                t_end),
              ylim=ylim, save=False)
