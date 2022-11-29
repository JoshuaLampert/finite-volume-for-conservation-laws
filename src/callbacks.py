import numpy as np
from matplotlib import pyplot as plt

class Callback:

    def __init__(self):
        pass

    def on_step_end(self, x, u, t):
        raise NotImplementedError()

    def on_end(self):
        raise NotImplementedError()

class PlotCallback(Callback):

    def __init__(self, ylim=None, analytic_sol=None):
        super().__init__()
        self.ylim = ylim
        if analytic_sol is not None:
            self.analytic_sol = np.vectorize(analytic_sol, otypes=[np.ndarray])
        else:
            self.analytic_sol = None

    def on_step_end(self, x, u, t):
        plt.ion()
        fig = plt.figure(1)
        plt.clf()
        #plt.plot(x, u[0, :], label="u[0]")
        plt.scatter(x, u[0, :], s=10, c="black", label="u[0]")
        if callable(self.analytic_sol):
            u_analytic = np.stack(self.analytic_sol(x, t)).T
            plt.plot(x, u_analytic[0, :], "orange",
                     label="analytical solution u[0]")
        plt.legend()
        plt.title("solution at time: {:.2f}".format(t))
        if self.ylim is not None:
            plt.ylim(self.ylim)
        plt.xlabel("x")
        plt.ylabel("u[0]")
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

    def on_step_end(self, x, u, t):
        u_analytic = np.stack(self.analytic_sol(x, t)).T
        self.errors.append(np.linalg.norm(u - u_analytic, self.error_type))
        self.ts.append(t)

    def on_end(self):
        fig = plt.figure()
        plt.semilogy(self.ts, self.errors,
                     label=str(self.error_type) + " error")
        plt.xlabel("t")
        plt.ylabel("log(error)")
        plt.legend()
        plt.show()
