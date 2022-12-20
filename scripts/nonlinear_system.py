import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.callbacks import PlotCallback
    from src.equations import NonlinearSystem
    from src.problem import Problem
    from src.util import plot_sols

    g = 1.0
    equation = NonlinearSystem()
    Nx, xmin, xmax = 100, -1.0, 1.0
    t_end = 0.2
    CFL = 0.95

    def Q0(x):
        if x < 0.0:
            return np.array([1.0, 0.5])
        else:
            return np.array([0.5, 1.0])

    ylim = [[0.0, 1.2], [0.0, 1.2]]
    bc = "transparent"
    callbacks = [PlotCallback(ylim=ylim, equation=equation, prim=False)]
    # callbacks = []
    problems = {}
    for num_flux in ["rusanov", "godunov"]:
        problem = Problem(Nx, xmin, xmax, t_end, equation=equation,
                          bc=bc, numerical_flux=num_flux, CFL=CFL,
                          callbacks=callbacks)
        problems[num_flux] = problem
    plot_sols(problems, Q0,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                Q0.__name__,
                                                                t_end),
              ylim=ylim, save=False)
