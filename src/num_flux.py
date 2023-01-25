import numpy as np
import sympy
from scipy import special

from .equations import Burgers, LinearScalar
from .reconstruction import WENOReconstruction
from .util import get_numerical_flux, IntegratorGL


class NumericalFlux:

    def __init__(self, equation=Burgers()):
        self.equation = equation

    def __call__(self, u_L, u_R):
        raise NotImplementedError()


class Rusanov(NumericalFlux):

    def __init__(self, equation=Burgers()):
        super().__init__(equation)

    def __call__(self, u_L, u_R):
        l_max = self.equation.max_eigenvalue(np.array([u_L, u_R]))
        return 0.5*(self.equation.flux(u_R) + self.equation.flux(u_L) -
                    l_max*(u_R - u_L))


class Roe(NumericalFlux):

    def __init__(self, equation=Burgers(), tol=1e-12):
        self.tol = tol
        super().__init__(equation)

    def __call__(self, u_L, u_R):
        if u_L.size > 1 or u_R.size > 1:
            raise NotImplementedError("Roe flux is only implemented for " +
                                      "scalar equations.")
        else:
            u_L = u_L[0]
            u_R = u_R[0]
            if np.abs(u_L - u_R) < self.tol:
                return self.equation.flux(u_L)
            else:
                F_L = self.equation.flux(u_L)
                F_R = self.equation.flux(u_R)
                s = (F_L - F_R)/(u_L - u_R)
                if s >= 0:
                    return np.array([F_L])
                else:
                    return np.array([F_R])


class Godunov(NumericalFlux):

    def __init__(self, equation=Burgers()):
        super().__init__(equation)

    def __call__(self, u_L, u_R):
        if self.equation.godunov_state is None:
            raise NotImplementedError("The equation has to implement the " +
                                      "godunov_state in order to use the " +
                                      "Godunov numerical_flux.")
        return self.equation.flux(self.equation.godunov_state(u_L, u_R))

class HLL(NumericalFlux):

    def __init__(self, equation=Burgers()):
        super().__init__(equation)

    def __call__(self, u_L, u_R):
        S_L, S_R = self.equation.min_max_speed(u_L, u_R)
        if S_L >= 0:
            return self.equation.flux(u_L)
        elif S_R <= 0:
            return self.equation.flux(u_R)
        else:
            F_L = self.equation.flux(u_L)
            F_R = self.equation.flux(u_R)
            F_HLL = (S_R*F_L - S_L*F_R + S_R*S_L*(u_R - u_L)) / (S_R - S_L)
            return F_HLL

class Eigen(NumericalFlux):

    def __init__(self, equation=Burgers(), C=0.5, numerical_flux='rusanov'):
        super().__init__(equation)
        self.C = C
        self.numerical_flux = get_numerical_flux(numerical_flux, equation)

    def __call__(self, u_L, u_R):
        l_max = self.equation.max_eigenvalue(np.array([u_L, u_R]))
        mu = self.C/l_max
        F_hat = self.numerical_flux(u_L, u_R)
        u_L_hat = u_L - mu*(F_hat - self.equation.flux(u_L))
        u_R_hat = u_R - mu*(self.equation.flux(u_R) - F_hat)
        A_hat = self.equation.flux_derivative(0.5*(u_L_hat + u_R_hat))
        lambda_hat, R_hat = np.linalg.eig(A_hat)
        alpha = np.linalg.solve(R_hat, u_R - u_L)
        u_star = 0.5*(u_L + u_R)
        for k in range(self.equation.m):
            u_star -= 0.5*np.sign(lambda_hat[k])*alpha[k]*R_hat[:, k]
        return self.equation.flux(u_star)

class SpaceBase:

    def __init__(self, N=3):
        self.N = N
        self.space_base = []
        x = sympy.symbols('x')
        for r in range(N + 1):
            # degree r polynomial
            poly = special.legendre(r)
            # transform Legendre polynomials from [0, 1] to [-1, 1]
            poly_sym = sympy.Poly(poly.c, x)
            poly_sym = sympy.compose(poly_sym, 2*x - 1)
            poly = np.poly1d(poly_sym.all_coeffs())

            # compute derivatives
            self.space_base.append([poly])
            for _ in range(N):
                poly = poly.deriv()
                # k-th derivative of degree r polynomial
                self.space_base[r].append(poly)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            phi = np.empty((self.N + 1, self.N + 1) + x.shape)
        else:
            phi = np.empty((self.N + 1, self.N + 1))
        for r, poly_deg_r in enumerate(self.space_base):
            for k, poly_deg_r_deriv_k in enumerate(poly_deg_r):
                phi[r, k] = np.polyval(poly_deg_r_deriv_k, x)
        return phi


class ADER(NumericalFlux):

    def __init__(self, mesh, equation=Burgers(), N=3, N_gl=8, bc="transparent"):
        self.mesh = mesh
        self.N = N
        self.N_gl = N_gl
        self.integrator = IntegratorGL(N_gl)
        self.weno = WENOReconstruction(N, bc=bc)
        space_base = SpaceBase(N)
        # phi_L and phi_R have shape (M, N + 1, N + 1, 1) where M number
        # unknowns. Last dimension to multiply with w_hat where last dimension
        # stores number of point in space
        self.phi_L = np.expand_dims(space_base(0.0), axis=(0, -1))
        self.phi_R = np.expand_dims(space_base(1.0), axis=(0, -1))
        super().__init__(equation)

    def prepare(self, u):
        w_hat = self.weno.reconstruct(u)
        self.du_dx_m = np.empty((u.shape[0], u.shape[1], self.N + 1))
        self.du_dx_p = np.empty((u.shape[0], u.shape[1], self.N + 1))
        # compute spatial derivatives of u
        for k in range(self.N + 1):
            self.du_dx_m[:, :, k] = 1/self.mesh.spatialmesh.dx ** k *\
                                    np.sum(w_hat*self.phi_L[:, :, k, :],
                                           axis=1)
            self.du_dx_p[:, :, k] = 1/self.mesh.spatialmesh.dx ** k *\
                np.sum(w_hat*self.phi_R[:, :, k, :], axis=1)
        # set new dt to integrator
        self.integrator.set_bounds(0, self.mesh.timemesh.dt)

    def __call__(self, j_L, j_R):
        if self.equation.cauchy_kovalevskaya is None:
            raise NotImplementedError("The equation has to implement " +
                                      "cauchy_kovalevskaya in order to use " +
                                      "the ADER numerical_flux.")
        # Solve generalized Riemann problems by Toro-Titarev solver
        du_dx_star = np.empty((self.du_dx_m.shape[0], self.N + 1))
        du_dx_star[:, 0] = self.equation.godunov_state(self.du_dx_p[:, j_L, 0],
                                                       self.du_dx_m[:, j_R, 0])
        # WARNING: Assume scalar equation here
        q0_star = self.equation.flux_derivative(du_dx_star[:, 0])[0]
        equation_linear = LinearScalar(q0_star)
        for k in range(self.N):
            du_dx_star[:, k + 1] = equation_linear.godunov_state(
                self.du_dx_p[:, j_L, k + 1],
                self.du_dx_m[:, j_R, k + 1])
        # compute time derivatives of u from spatial derivatives
        du_dt = self.equation.cauchy_kovalevskaya(du_dx_star)
        # approximation to f(u(x_{j\pm 1/2}, 0_+))

        def f_u(tau):
            # u_taylor = 0.0
            # for k in range(self.N + 1):
            #    u_taylor += du_dt[:, k] * tau**k / np.math.factorial(k)
            zero_N = np.arange(self.N + 1)
            u_taylor = np.sum(du_dt * tau**zero_N / special.factorial(zero_N),
                              axis=1)
            return self.equation.flux(u_taylor)
        return 1/self.mesh.timemesh.dt * self.integrator.integrate(f_u)
