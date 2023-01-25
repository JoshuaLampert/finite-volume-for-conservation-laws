import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.mesh import Mesh
    from src.callbacks import StepsizeCallback, PlotCallback
    from src.equations import Euler
    from src.problem import Problem
    from src.util import plot_sols

    case = 6
    if case == 0:
        gamma = 1.4
        t_end = 0.2
        ylim = [[0.0, 1.2], [-0.2, 1.0], [0.0, 1.2], [1.7, 3.0]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([1.0, 0.0, 1.0])
            else:
                return np.array([0.125, 0.0, 0.1])
    elif case == 1:
        gamma = 1.4
        t_end = 0.13
        ylim = [[0.2, 1.2], [-0.1, 1.8], [0.0, 4.0], [2.0, 22.0]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([0.445, 0.698, 3.528])
            else:
                return np.array([0.5, 0.0, 0.571])
    elif case == 2:
        gamma = 5/3
        t_end = 0.4
        ylim = [[0.0, 6.0], [-2.5, 2.5], [0.0, 8.0], [0.1, 3.0]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([1.0, 2.0, 0.2])
            else:
                return np.array([1.5, -2.0, 0.2])
    elif case == 3:
        gamma = 5/3
        t_end = 0.08
        ylim = [[0.1, 1.8], [-3.0, 3.0], [0.0, 4.5], [1.5, 4.5]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([1.0, -2.5, 2.0])
            else:
                return np.array([1.5, 2.5, 4.0])
    elif case == 4:
        gamma = 1.4
        t_end = 0.47
        ylim = [[0.0, 5.0], [-1.0, 6.0], [0.0, 15.0], [2.0, 7.5]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([3.57134, 2.629369, 10.33333])
            else:
                return np.array([1.0 + 0.2*np.sin(5*np.pi*x), 0.0, 1.0])
    elif case == 5:
        gamma = 1.4
        t_end = 0.5
        ylim = [[0.8, 1.5], [-2.0, 2.0], [0.2, 1.8], [1.7, 2.6]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([1.4, 0.0, 1.0])
            else:
                return np.array([1.0, 0.0, 1.0])
    elif case == 6:
        gamma = 1.4
        t_end = 0.2
        ylim = [[0.0, 1.2], [-0.2, 1.5], [0.0, 1.2], [1.9, 3.5]]

        def u0_prim(x):
            if x < 0.0:
                return np.array([1.0, 0.74, 1.0])
            else:
                return np.array([0.125, 0.0, 0.1])
    equation = Euler(gamma)

    def u0(x):
        return equation.prim2cons(u0_prim(x))
    xmin, xmax, Nx = -1.0, 1.0, 100
    mesh = Mesh(xmin, xmax, 0.0, t_end, Nx, dt=1.0)
    CFL = 0.95

    bc = "transparent"
    callbacks = [StepsizeCallback(equation, mesh, CFL=CFL),
                 PlotCallback(equation,
                              additional_plots=[equation.internal_energy_cons],
                              ylim=ylim)]
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
              additional_plots=[equation.internal_energy_cons],
              ylim=ylim, save=False)
