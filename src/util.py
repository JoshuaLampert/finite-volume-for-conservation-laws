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
    for key, u in sols.items():
        #plt.plot(problems[key].x, u[0, :], label=key)
        plt.scatter(problems[key].x, u[0, :], s=10, label=key)
    if analytic_sol is not None:
        if callable(analytic_sol):
            sol = np.vectorize(analytic_sol, otypes=[np.ndarray])
            x = problems[key].x
            u = np.stack(sol(x)).T
            plt.plot(x, u[0, :], "orange", label="analytical solution")
        else:
            raise NotImplementedError("Analytical solution has to be " + \
                                      "provided as callable function")
    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("u[0]")
    if ylim is not None:
            plt.ylim(ylim)
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
               error_type=np.inf):
    analytic_sol = np.vectorize(analytic_sol, otypes=[np.ndarray])
    errors = {}
    orders = {}
    dNx = np.diff(np.log(Nxs))
    for key, problem in problems.items():
        errors[key] = []
        for Nx in Nxs:
            problem.set_Nx(Nx)
            start = time.time()
            u = problem.solve(g)[-1][0, :]
            end = time.time()
            print("finished {} for Nx={} in {} s".format(key, Nx, end - start))
            u_analytic = np.stack(analytic_sol(problem.x, problem.t_end))[:, 0]
            error = np.linalg.norm(u - u_analytic, error_type)
            errors[key].append(error)
        diff = -np.diff(np.log(errors[key]))
        orders[key] = diff / dNx
    print(orders)
    print(errors)
    plt.figure()
    for key, error in errors.items():
        plt.loglog(Nxs, error, label=key)
    for i in range(1, 8):
        plt.loglog(Nxs, 2/Nxs**i, '--', c='gray')
    plt.legend()
    plt.xlabel("log(Nx)")
    plt.ylabel("log(error)")
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
