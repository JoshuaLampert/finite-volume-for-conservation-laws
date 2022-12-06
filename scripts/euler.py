import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.callbacks import PlotCallback
    from src.equations import Euler
    from src.problem import Problem
    from src.util import plot_sols

    case = 4
    if case == 0:
        gamma = 1.4
        t_end = 0.2
        ylim = [[0.0, 1.2], [-0.2, 1.0], [0.0, 1.2]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([1.0, 0.0, 1.0])
            else:
                return np.array([0.125, 0.0, 0.1])
    elif case == 1:
        gamma = 1.4
        t_end = 0.13
        ylim = [[0.2, 1.2], [-0.1, 1.8], [0.0, 4.0]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([0.445, 0.698, 3.528])
            else:
                return np.array([0.5, 0.0, 0.571])
    elif case == 2:
        gamma = 5 / 3
        t_end = 0.4
        ylim = [[0.0, 6.0], [-2.5, 2.5], [0.0, 8.0]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([1.0, 2.0, 0.2])
            else:
                return np.array([1.5, -2.0, 0.2])
    elif case == 3:
        gamma = 5/3
        t_end = 0.08
        ylim = [[0.1, 1.8], [-3.0, 3.0], [0.0, 4.5]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([1.0, -2.5, 2.0])
            else:
                return np.array([1.5, 2.5, 4.0])
    elif case == 4:
        gamma = 1.4
        t_end = 0.47
        ylim = [[0.0, 5.0], [-1.0, 6.0], [0.0, 15.0]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([3.57134, 2.629369, 10.33333])
            else:
                return np.array([1.0 + 0.2*np.sin(5*np.pi*x), 0.0, 1.0])
    equation = Euler(gamma)

    def u0(x):
        return equation.prim2cons(u0_prim(x))
    Nx, xmin, xmax = 100, -1.0, 1.0
    CFL = 0.95

    bc = "transparent"
    callbacks = [PlotCallback(ylim=ylim, equation=equation)]
    # callbacks = []
    problems = {}
    for num_flux in ["rusanov"]:
        problem = Problem(Nx, xmin, xmax, t_end, equation=equation,
                          bc=bc, numerical_flux=num_flux, CFL=CFL,
                          callbacks=callbacks)
        problems[num_flux] = problem
    plot_sols(problems, u0,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                u0.__name__,
                                                                t_end),
              ylim=ylim, save=False)
