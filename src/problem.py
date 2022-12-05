import numpy as np
from .equations import Burgers
from .num_flux import NumericalFlux, Rusanov, LaxWendroff, Roe, Godunov, ADER
from .util import boundary_condition, integrate_gl

class Problem:

    def __init__(self, Nx, x0, x1, t_end, equation=Burgers(), bc="transparent",
                 numerical_flux='rusanov', N=3, CFL=0.95, Nt_max=int(1e5),
                 N_gl=8, callbacks=[]):
        self.Nx = Nx
        self.x0 = x0
        self.x1 = x1
        dx = (x1 - x0)/Nx
        self.dx = dx
        self.x = np.linspace(x0 + dx/2, x1 - dx/2, Nx)
        self.t_end = t_end

        self.equation = equation
        self.bc = bc
        # number of points of Gauss-Legendre quadrature
        self.N_gl = N_gl
        if isinstance(numerical_flux, NumericalFlux):
            self.numerical_flux = numerical_flux
        elif numerical_flux == 'rusanov':
            self.numerical_flux = Rusanov(equation)
        elif numerical_flux == 'LxW':
            # first, set some dt (but will be overwritten by CFL cond. later)
            self.numerical_flux = LaxWendroff(dx, 1.0, equation)
        elif numerical_flux == 'godunov':
            self.numerical_flux = Godunov(equation)
        elif numerical_flux == 'roe':
            self.numerical_flux = Roe(equation)
        elif numerical_flux == 'ader':
            # first, set some dt (but will be overwritten by CFL cond. later)
            self.numerical_flux = ADER(self.x, 1.0, dx, N, N_gl, bc, equation)
        else:
            raise NotImplementedError("Unknown numerical_flux {}.".format(
                numerical_flux) + " Implemented are 'rusanov', 'LxW', 'roe'" + \
                                      ", 'godunov' and 'ader'.")
        self.CFL = CFL
        self.Nt_max = Nt_max
        self.callbacks = callbacks

    def set_Nx(self, Nx):
        self.Nx = Nx
        self.dx = (self.x1 - self.x0)/Nx
        self.x = np.linspace(self.x0 + self.dx/2, self.x1 - self.dx/2, Nx)
        if isinstance(self.numerical_flux, LaxWendroff) or \
           isinstance(self.numerical_flux, ADER):
            self.numerical_flux.dx = self.dx

    def solve(self, g):
        # evaluate initial value just at cell centers
        # g = np.vectorize(g, otypes=[np.ndarray])
        # u0 = np.stack(g(self.x)).T
        # average initial value over cells (more precise)
        num_unkn = g(0.0).shape[0]
        u0 = np.empty((num_unkn, self.Nx))
        for j in range(self.Nx):
            u0[:, j] = 1/self.dx*integrate_gl(g, self.x[j] - self.dx/2,
                                              self.x[j] + self.dx/2, self.N_gl)
            u = [u0]
        t = 0
        for callback in self.callbacks:
            callback.on_step_end(self.x, u[-1], t)
        for i in range(self.Nt_max):
            u_new, dt = self.step(u[-1], t)
            u.append(u_new)
            t += dt
            for callback in self.callbacks:
                callback.on_step_end(self.x, u[-1], t)
            if t >= self.t_end:
                for callbak in self.callbacks:
                    callback.on_end()
                return np.array(u)
        for callbak in self.callbacks:
            callback.on_end()
        return np.array(u)

    def step(self, u, t):
        lmax = 0.0
        for j in range(self.Nx):
            lambda_ = np.max(np.abs(self.equation.eigenvalues(u[:, j])))
            lmax = np.maximum(lmax, lambda_)
        dx = self.dx
        dt = self.CFL*dx/lmax
        if t + dt > self.t_end:
            dt = self.t_end - t
        self.numerical_flux.dt = dt
        if isinstance(self.numerical_flux, ADER):
            self.numerical_flux.prepare(u)
        u_new = np.empty(u.shape)
        for j in range(self.Nx):
            j_m = boundary_condition(j - 1, self.Nx, self.bc)
            j_p = boundary_condition(j + 1, self.Nx, self.bc)
            if isinstance(self.numerical_flux, ADER):
                F_L = self.numerical_flux(j_m, j)
                F_R = self.numerical_flux(j, j_p)
            else:
                F_L = self.numerical_flux(u[:, j_m], u[:, j])
                F_R = self.numerical_flux(u[:, j], u[:, j_p])
            u_new[:, j] = u[:, j] - dt/dx*(F_R - F_L)
        return u_new, dt
