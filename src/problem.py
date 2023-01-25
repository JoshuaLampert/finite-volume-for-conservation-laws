import numpy as np

from .equations import Burgers
from .num_flux import ADER, Godunov, NumericalFlux, Roe, Rusanov
from .util import boundary_condition, contains_stepsize_callback, integrate_gl


class Problem:

    def __init__(self, mesh, equation=Burgers(), bc="transparent",
                 numerical_flux='rusanov', N=3, Nt_max=int(1e5),
                 N_gl=8, callbacks=[]):
        self.mesh = mesh
        self.equation = equation
        self.bc = bc
        # number of points of Gauss-Legendre quadrature
        self.N_gl = N_gl
        if isinstance(numerical_flux, NumericalFlux):
            self.numerical_flux = numerical_flux
        elif numerical_flux == 'rusanov':
            self.numerical_flux = Rusanov(equation)
        elif numerical_flux == 'godunov':
            self.numerical_flux = Godunov(equation)
        elif numerical_flux == 'roe':
            self.numerical_flux = Roe(equation)
        elif numerical_flux == 'ader':
            # first, set some dt (but will be overwritten by CFL cond. later)
            self.numerical_flux = ADER(self.mesh, N, N_gl, bc, equation)
        else:
            raise NotImplementedError("Unknown numerical_flux {}.".format(
                numerical_flux) + " Implemented are 'rusanov', 'roe'" +
                                      ", 'godunov' and 'ader'.")
        self.Nt_max = Nt_max
        self.callbacks = callbacks

    def set_Nx(self, Nx):
        xmax = self.mesh.spatialmesh.xmax
        xmin = self.mesh.spatialmesh.xmin
        self.mesh.spatialmesh.Nx = Nx
        dx = (xmax - xmin)/Nx
        self.mesh.spatialmesh.dx = dx
        self.mesh.spatialmesh.x = np.linspace(xmin + dx/2, xmax - dx/2, Nx)
        if isinstance(self.numerical_flux, ADER):
            self.numerical_flux.mesh = self.mesh

    def solve(self, g):
        # evaluate initial value just at cell centers
        # g = np.vectorize(g, otypes=[np.ndarray])
        # u0 = np.stack(g(self.x)).T
        # average initial value over cells (more precise)
        num_unkn = self.equation.m
        self.mesh.reset()
        x = self.mesh.spatialmesh.x
        dx = self.mesh.spatialmesh.dx
        Nx = self.mesh.spatialmesh.Nx
        u0 = np.empty((num_unkn, Nx))
        for j in range(Nx):
            u0[:, j] = 1/dx*integrate_gl(g, x[j] - dx/2, x[j] + dx/2, self.N_gl)
        u = [u0]
        for n in range(self.Nt_max):
            if not contains_stepsize_callback(self.callbacks):
                self.mesh.update()
            for callback in self.callbacks:
                callback.on_step_begin(x, u[n], self.mesh.timemesh.t[n])
            u_new = self.step(u[-1])
            u.append(u_new)
            for callback in self.callbacks:
                callback.on_step_end(x, u[n + 1], self.mesh.timemesh.t[n + 1])
            if self.mesh.isfinished():
                for callback in self.callbacks:
                    callback.on_end()
                return np.array(u)
        for callback in self.callbacks:
            callback.on_end()
        return np.array(u)

    def step(self, u):
        dx = self.mesh.spatialmesh.dx
        dt = self.mesh.timemesh.dt
        if isinstance(self.numerical_flux, ADER):
            self.numerical_flux.prepare(u)
        u_new = np.empty(u.shape)
        for j in range(self.mesh.spatialmesh.Nx):
            j_m = boundary_condition(j - 1, self.mesh.spatialmesh.Nx, self.bc)
            j_p = boundary_condition(j + 1, self.mesh.spatialmesh.Nx, self.bc)
            if isinstance(self.numerical_flux, ADER):
                F_L = self.numerical_flux(j_m, j)
                F_R = self.numerical_flux(j, j_p)
            else:
                F_L = self.numerical_flux(u[:, j_m], u[:, j])
                F_R = self.numerical_flux(u[:, j], u[:, j_p])
            u_new[:, j] = u[:, j] - dt/dx*(F_R - F_L)
        return u_new
