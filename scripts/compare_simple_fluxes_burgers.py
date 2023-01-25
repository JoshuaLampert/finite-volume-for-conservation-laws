import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.mesh import Mesh
    from src.callbacks import StepsizeCallback, PlotCallback
    from src.equations import Burgers
    from src.problem import Problem
    from src.util import plot_sols

    equation = Burgers()
    t_end = 5.0
    xmin, xmax, Nx = -2.0, 2.0, 100
    mesh = Mesh(xmin, xmax, 0.0, t_end, Nx, dt=1.0)
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
    ylim = [[-1.0, 1.0]]
    callbacks = [StepsizeCallback(equation, mesh, CFL=CFL),
                 PlotCallback(equation, ylim=ylim)]
    # callbacks = [StepsizeCallback(equation, mesh, CFL=CFL)]
    problems = {}
    for num_flux in ['rusanov', 'godunov', 'roe']:
        problem = Problem(mesh, equation=equation, bc=bc,
                          numerical_flux=num_flux, callbacks=callbacks)
        problems[num_flux] = problem
    plot_sols(problems, g,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                g.__name__,
                                                                t_end),
              ylim=ylim, save=False)
