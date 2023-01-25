import numpy as np
from matplotlib import pyplot as plt


class Callback:

    def __init__(self):
        pass

    def on_step_begin(self, x, u, t):
        raise NotImplementedError()

    def on_step_end(self, x, u, t):
        raise NotImplementedError()

    def on_end(self):
        raise NotImplementedError()


class PlotCallback(Callback):

    def __init__(self, equation, additional_plots=[], ylim=None,
                 analytic_sol=None, prim=True):
        super().__init__()
        self.equation = equation
        self.additional_plots = additional_plots
        self.ylim = ylim
        if analytic_sol is not None:
            self.analytic_sol = np.vectorize(analytic_sol, otypes=[np.ndarray])
        else:
            self.analytic_sol = None
        self.prim = prim

    def on_step_begin(self, x, u, t):
        pass

    def on_step_end(self, x, u, t):
        m = self.equation.m
        num_plots = m + len(self.additional_plots)
        plt.ion()
        fig = plt.figure(1)
        plt.clf()
        name = "U"
        if callable(self.analytic_sol):
            u_analytic = np.stack(self.analytic_sol(x, t)).T
        if self.prim:
            try:
                qu = self.equation.cons2prim(u)
                name = "Q"
                if callable(self.analytic_sol):
                    qu_analytic = self.equation.cons2prim(u_analytic)
            except AttributeError:
                qu = u
                qu_analytic = u_analytic
                # print("No primitive variables defined. Plot conservative.")

        for i in range(m):
            ax = plt.subplot(1, num_plots, i + 1)
            ax.scatter(x, qu[i, :], s=10, c="black", label="{}[{}]".format(name,
                                                                          i))
            # ax.plot(x, qu[i, :], label="{}[{}]".format(name, i))
            if callable(self.analytic_sol):
                plt.plot(x, u_analytic[i, :], "orange",
                         label="analytical solution {}[{}]".format(name, i))
            ax.legend()
            ax.set(xlabel="x", ylabel="{}[{}]".format(name, i),
                   title="{}[{}]".format(name, i))
            if isinstance(self.ylim, list):
                if self.ylim[i] is not None:
                    ax.set(ylim=self.ylim[i])
        for i in range(len(self.additional_plots)):
            func = self.additional_plots[i]
            ax = plt.subplot(1, num_plots, i + m + 1)
            ax.scatter(x, func(u), s=10, c="black", label=func.__name__)
            if callable(self.analytic_sol):
                plt.plot(x, func(u_analytic), "orange",
                         label="analytical solution {}".format(func.__name__))
            ax.legend()
            ax.set(xlabel="x", ylabel=func.__name__, title=func.__name__)
            if isinstance(self.ylim, list):
                if self.ylim[m + i] is not None:
                    ax.set(ylim=self.ylim[m + i])
        plt.suptitle("solution at time: {:.2f}".format(t))
        fig.canvas.draw()
        fig.canvas.flush_events()

    def on_end(self):
        pass


class ErrorCallback(Callback):

    def __init__(self, analytic_sol, error_type=2):
        super().__init__()
        self.analytic_sol = np.vectorize(analytic_sol, otypes=[np.ndarray])
        self.error_type = error_type
        self.errors = []
        self.ts = []

    def on_step_begin(self, x, u, t):
        pass

    def on_step_end(self, x, u, t):
        u_analytic = np.stack(self.analytic_sol(x, t)).T
        self.errors.append(np.linalg.norm(u - u_analytic, self.error_type))
        self.ts.append(t)

    def on_end(self):
        plt.figure()
        plt.semilogy(self.ts, self.errors,
                     label=str(self.error_type) + " error")
        plt.xlabel("t")
        plt.ylabel("log(error)")
        plt.legend()
        plt.show()

class StepsizeCallback(Callback):

    def __init__(self, equation, mesh, CFL=0.95):
        self.equation = equation
        self.mesh = mesh
        self.CFL = CFL

    def on_step_begin(self, x, u, t):
        dx = self.mesh.spatialmesh.dx
        l_max = self.equation.max_eigenvalue(u.T)
        dt = self.CFL * dx / l_max
        self.mesh.update(dt)

    def on_step_end(self, x, u, t):
        pass

    def on_end(self):
        pass
