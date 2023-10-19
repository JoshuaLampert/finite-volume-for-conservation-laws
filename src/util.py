import time

import numpy as np
from matplotlib import pyplot as plt

from .equations import Burgers


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


def contains_stepsize_callback(callbacks):
    from .callbacks import StepsizeCallback
    for callback in callbacks:
        if isinstance(callback, StepsizeCallback):
            return True
    return False


def get_numerical_flux(numerical_flux, equation=Burgers(),
                       mesh=None, N=3, N_gl=8, bc="transparent"):
    from .num_flux import NumericalFlux, Rusanov, Roe, Godunov, HLL, PredCorr
    from .num_flux import ADER
    if isinstance(numerical_flux, NumericalFlux):
        return numerical_flux
    elif numerical_flux.lower() == 'rusanov':
        return Rusanov(equation)
    elif numerical_flux.lower() == 'roe':
        return Roe(equation)
    elif numerical_flux.lower() == 'godunov':
        return Godunov(equation)
    elif numerical_flux.lower() == 'hll':
        return HLL(equation)
    elif numerical_flux.lower() == 'predcorr':
        return PredCorr(equation)
    elif numerical_flux.lower() == 'ader':
        if mesh is None:
            raise ValueError("You need to provide a mesh to use the ADER flux")
        return ADER(mesh, equation, N, N_gl, bc)
    else:
        raise NotImplementedError("Unknown numerical_flux {}.".format(
            numerical_flux) + " Implemented are 'rusanov', 'roe'" +
                                  ", 'godunov', 'hll', 'predcorr' and 'ader'.")


def plot_sols(problems, g, title="", additional_plots=[], ylim=None, save=True,
              analytic_sol=None, prim=True):
    sols = {}
    for key, problem in problems.items():
        start = time.time()
        sols[key] = problem.solve(g)[-1]
        end = time.time()
        print("solved {} in {} s".format(key, end - start))
    plt.clf()
    m = list(problems.values())[0].equation.m
    num_plots = m + len(additional_plots)
    for key, u in sols.items():
        x = problems[key].mesh.spatialmesh.x
        for i in range(m):
            ax = plt.subplot(1, num_plots, i + 1)
            name = "U"
            if prim:
                try:
                    qu = problems[key].equation.cons2prim(u)
                    name = "Q"
                except AttributeError:
                    qu = u
                    # print("No primitive variables defined. Plot " +
                    # "conservative.")
            # ax.plot(problems[key].x, u[i, :], label=key)
            ax.scatter(x, qu[i, :], s=10, label=key)
            ax.legend()
            ax.set(xlabel="x", ylabel="{}[{}]".format(name, i),
                   title="{}[{}]".format(name, i))
            if isinstance(ylim, list):
                if ylim[i] is not None:
                    ax.set(ylim=ylim[i])
        if analytic_sol is not None:
            if callable(analytic_sol):
                sol = np.vectorize(analytic_sol, otypes=[np.ndarray])
                u = np.stack(sol(x)).T
                if prim:
                    try:
                        qu = problems[key].equation.cons2prim(u)
                    except AttributeError:
                        qu = u
                        # print("No primitive variables defined. Plot " + \
                        #      "conservative.")

                ax = plt.subplot(1, m, i + 1)
                ax.plot(x, qu[i, :], "orange", label="analytical solution")
            else:
                raise NotImplementedError("Analytical solution has to be " +
                                          "provided as callable function")
        ax.legend()
        for i in range(len(additional_plots)):
            func = additional_plots[i]
            ax = plt.subplot(1, num_plots, i + m + 1)
            ax.scatter(x, func(u), s=10, label=key)
            if callable(analytic_sol):
                plt.plot(x, func(u_analytic), "orange",
                         label="analytical solution {}".format(func.__name__))
            ax.legend()
            ax.set(xlabel="x", ylabel=func.__name__, title=func.__name__)
            if isinstance(ylim, list):
                if ylim[m + i] is not None:
                    ax.set(ylim=ylim[m + i])
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
            u_ana = lambda x: analytic_sol(x, problem.mesh.timemesh.t_end)
            u_ana_vec = np.empty((u.shape[0], Nx))
            x = problem.mesh.spatialmesh.x
            dx = problem.mesh.spatialmesh.dx
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
        # gx[:, i] = g(xi[i]*(b - a) + a) # for xi in [0,1]
    return np.sum(w*gx, axis=1)
