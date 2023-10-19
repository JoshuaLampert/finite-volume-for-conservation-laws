import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from src.mesh import Mesh
    from src.callbacks import StepsizeCallback, PlotCallback
    from src.equations import Burgers
    from src.problem import Problem
    from src.util import plot_sols

    xmin, xmax, Nx = -1.0, 1.0, 100
    t_end = 0.25
    mesh = Mesh(xmin, xmax, 0.0, t_end, Nx, dt=1.0)
    a = 1.0
    # equation = LinearScalar()
    equation = Burgers()

    def g1(x):
        if x < 0:
            return np.array([x + 1])
        else:
            return np.array([-2*x + 2])

    def sol1(x, t):
        if x < t:
            return np.array([(x - t)/(1 + t) + 1])
        elif x > 2*t:
            return np.array([-2*(x - 2*t)/(1 - 2*t) + 2])
        else:
            return np.array([x/t])

    def g2(x):
        return np.array([8*np.exp(-40*(x - 0.5)**2)*np.sin(16*np.pi*x)])

    def g3(x):
        return np.array([np.sin(2*np.pi*x)**2])

    def g4(x):
        return np.array([(1 - x)**2*(1 + x)**2*np.exp(-np.sin(6*np.pi*x))])
    g = g4
    bc = "periodic"

    def sol_linear(x, t):
        if bc == "transparent":
            return g(x - a*t)
        elif bc == "periodic":
            return g((x - a*t - xmin) % (xmax - xmin) + xmin)
        else:
            raise NotImplementedError()
    if equation.name == "linear advection equation":
        sol = sol_linear
    elif equation.name == "Burgers equation" and g == g1:
        sol = sol1
    else:
        sol = None
    ylim = [[0.0, 3.0]]
    callbacks = [StepsizeCallback(equation, mesh, CFL=0.95),
                 PlotCallback(equation, ylim=ylim, analytic_sol=sol)]
    # callbacks = [StepsizeCallback(equation, mesh, CFL=0.95)]
    N_gl = 8
    problems = {}
    for N in range(1, 7):
        problem_ader = Problem(mesh, equation=equation, bc=bc,
                               numerical_flux='ader', N=N,
                               Nt_max=int(1e9), N_gl=N_gl, callbacks=callbacks)
        problems["ADER" + str(N)] = problem_ader
    if callable(sol):
        def analytic_sol(x): return sol(x, t=t_end)
    else:
        analytic_sol = None
    plot_sols(problems, g,
              title="{} with initial data {} at time {}".format(equation.name,
                                                                g.__name__,
                                                                t_end),
              ylim=ylim, save=False, analytic_sol=analytic_sol)
    # compare_times(problems, g)
