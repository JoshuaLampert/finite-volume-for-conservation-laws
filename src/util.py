import numpy as np
from matplotlib import pyplot as plt
import time

def boundary_condition(j, Nx, bc="periodic"):
    if bc == "periodic":
        if j < 0:
            return j + Nx
        elif j >= Nx:
            return j - Nx
        else:
            return j
    elif bc == "transparent":
        if j < 0:
            return -j - 1
        elif j >= Nx:
            return Nx - (j - Nx) - 1
        else:
            return j
    else:
        raise NotImplementedError("Unknown boundary_condition {}.".format(
            bc) + " Implemented are 'periodic' and 'transparent'.")

def plot_sols(problems, g, title="", ylim=None, save=True, analytic_sol=None):
    sols = {}
    for key, problem in problems.items():
        start = time.time()
        sols[key] = problem.solve(g)[-1]
        end = time.time()
        print("solved {} in {} s".format(key, end - start))
    plt.clf()
    m = list(problems.values())[0].equation.m
    for i in range(m):
        ax = plt.subplot(1, m, i + 1)
        for key, u in sols.items():
            #ax.plot(problems[key].x, u[i, :], label=key)
            ax.scatter(problems[key].x, u[i, :], s=10, label=key)
            ax.legend()
            ax.set(xlabel="x", ylabel="u[{}]".format(i),
                   title="u[{}]".format(i))
            if ylim is not None:
                ax.set(ylim=ylim[i])
        if analytic_sol is not None:
            if callable(analytic_sol):
                sol = np.vectorize(analytic_sol, otypes=[np.ndarray])
                x = problems[key].x
                u = np.stack(sol(x)).T
                ax = plt.subplot(1, m, i + 1)
                ax.plot(x, u[i, :], "orange", label="analytical solution")
            else:
                raise NotImplementedError("Analytical solution has to be " + \
                                          "provided as callable function")
    plt.suptitle(title)
    if save:
        plt.savefig(title + ".jpg")
    else:
        plt.show()

def compare_times(problems, g):
    for key, problem in problems.items():
        start = time.time()
        problem.solve(g)
        comp_time = time.time() - start
        print("solved problem {} in {} s".format(key, comp_time))

def plot_order(problems, g, analytic_sol, Nxs=16*2**np.arange(5),
               error_type=np.inf, save=False):
    errors = {}
    orders = {}
    dNx = np.diff(np.log(Nxs))
    for key, problem in problems.items():
        errors[key] = []
        for Nx in Nxs:
            problem.set_Nx(Nx)
            start = time.time()
            u = problem.solve(g)[-1]
            end = time.time()
            print("finished {} for Nx={} in {} s".format(key, Nx, end - start))
            u_ana = lambda x: analytic_sol(x, problem.t_end)
            u_ana_vec = np.empty((u.shape[0], Nx))
            x = problem.x
            dx = problem.dx
            for j in range(Nx):
                u_ana_vec[:, j] = integrate_gl(u_ana, x[j] - dx/2, x[j] + dx/2)
                u_ana_vec[:, j] /= dx

            # only for scalar problems:
            error = np.linalg.norm(u[0, :] - u_ana_vec[0, :], error_type)
            if not error_type == np.inf:
                error /= Nx**(1/error_type)
            errors[key].append(error)
        diff = -np.diff(np.log(errors[key]))
        orders[key] = diff / dNx
    print(orders)
    print(errors)
    plt.figure()
    for key, error in errors.items():
        plt.loglog(Nxs, error, label=key)
    for i in range(1, 8):
        plt.loglog(Nxs, 1/Nxs**i, '--', c='gray')
    plt.legend()
    plt.xlabel("Nx")
    plt.ylabel("error")
    if save:
        plt.savefig("img/orders_{}.jpg".format(error_type))
    else:
        plt.show()

class IntegratorGL:

    def __init__(self, N_gl=8, a=0.0, b=1.0):
        xi, w = np.polynomial.legendre.leggauss(N_gl)
        self.xi = 0.5*xi*(b - a) + 0.5*(b + a)
        self.w = w*(b - a)/2
        self.xi_norm = xi
        self.w_norm = w

    def set_bounds(self, a, b):
        self.xi = 0.5*self.xi_norm*(b - a) + 0.5*(b + a)
        self.w = self.w_norm*(b - a)/2

    def integrate(self, g):
        gx = np.vectorize(g)(self.xi)
        return np.sum(self.w*gx)
        
def integrate_gl(g, a, b, N_gl=8):
    """Gauss-Legendre quadrature."""
    # xi in [-1,1]
    xi, w = np.polynomial.legendre.leggauss(N_gl)
    w *= (b - a)/2
    xi = 0.5*xi*(b - a) + 0.5*(b + a)
    num_unkn = g(0.0).shape[0]
    gx = np.empty((num_unkn, xi.size))
    for i in range(xi.size):
        gx[:, i] = g(xi[i])
        #gx[:, i] = g(xi[i]*(b - a) + a) # for xi in [0,1]
    return np.sum(w*gx, axis=1)
