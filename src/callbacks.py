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

    def __init__(self, ylim=None, analytic_sol=None, equation=None, prim=True):
        super().__init__()
        self.ylim = ylim
        if analytic_sol is not None:
            self.analytic_sol = np.vectorize(analytic_sol, otypes=[np.ndarray])
        else:
            self.analytic_sol = None
        self.equation = equation
        self.prim = prim

    def on_step_end(self, x, u, t):
        m = u.shape[0]
        plt.ion()
        fig = plt.figure(1)
        plt.clf()
        name = "U"
        if callable(self.analytic_sol):
            u_analytic = np.stack(self.analytic_sol(x, t)).T
        if self.prim:
            try:
                u = self.equation.cons2prim(u)
                name = "Q"
                if callable(self.analytic_sol):
                    u_analytic = self.equation.cons2prim(u_analytic)
            except:
                pass
                #print("No primitive variables defined. Plot conservative.")

        for i in range(m):
            ax = plt.subplot(1, m, i + 1)
            ax.scatter(x, u[i, :], s=10, c="black", label="{}[{}]".format(name,
                                                                          i))
            #ax.plot(x, u[i, :], label="{}[{}]".format(name, i))
            if callable(self.analytic_sol):
                plt.plot(x, u_analytic[i, :], "orange",
                         label="analytical solution {}[{}]".format(name, i))
            ax.legend()
            ax.set(xlabel="x", ylabel="{}[{}]".format(name, i),
                   title="{}[{}]".format(name, i))
            if self.ylim is not None:
                ax.set(ylim=self.ylim[i])
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
