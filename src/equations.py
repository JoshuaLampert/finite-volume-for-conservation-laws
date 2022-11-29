import numpy as np

class Equation:

    def __init__(self, name):

        self.name = name

    """flux and flux_derivative are functions that take a vector u and return
    a vector of same length (number of unknowns / equations) for the flux
    and a square matrix for the flux_derivative"""
    def flux(self, u):
        raise NotImplementedError()

    def flux_derivative(self, u, k=1):
        raise NotImplementedError()

    """godunov_state is only needed for applying the Godunov method or the
    ADER flux to this equation"""
    def godunov_state(self, u_L, u_R):
        raise NotImplementedError()

    """cauchy_kovalevskaya is only needed for applying the ADER flux to this
    equation"""
    def cauchy_kovalevskaya(self, du_dx):
        N = du_dx.shape[1] - 1
        u = du_dx[:, 0]
        df = np.empty(du_dx.shape)
        for k in range(N + 1):
            df[:, k] = self.flux_derivative(u, k)
        du_dt = np.empty(du_dx.shape)
        du_dt[:, 0] = u
        if N >= 1:
            u_x = du_dx[:, 1]
            u_t = -df[:, 1] * u_x
            du_dt[:, 1] = u_t
        if N >= 2:
            u_xx = du_dx[:, 2]
            u_xt = -df[:, 1]*u_xx - df[:, 2]*u_x**2
            u_tt = -df[:, 1]*u_xt - df[:, 2]*u_t*u_x
            du_dt[:, 2] = u_tt
        if N >= 3:
            u_xxx = du_dx[:, 3]
            u_xxt = -df[:, 3]*u_x**3 - df[:, 1]*u_xxx - 3*df[:, 2]*u_x*u_xx
            u_xtt = -df[:, 1]*u_xxt - df[:, 2]*u_t*u_xx - 2*df[:, 2]*u_x*u_xt -\
                    df[:, 3]*u_t*u_x**2
            u_ttt = -df[:, 1]*u_xtt - 2*df[:, 2]*u_t*u_xt - df[:, 2]*u_x*u_tt -\
                    df[:, 3]*u_t**2*u_x
            du_dt[:, 3] = u_ttt
        if N >= 4:
            u_xxxx = du_dx[:, 4]
            u_xxxt = -df[:, 1]*u_xxxx - 3*df[:, 2]*u_xx**2 - df[:, 4]*u_x**4 - \
                     4*df[:, 2]*u_x*u_xxx - 6*df[:, 3]*u_x**2*u_xx
            u_xxtt = -df[:, 1]*u_xxxt - df[:, 4]*u_t*u_x**3 - \
                     df[:, 2]*u_t*u_xxx - 3*df[:, 2]*u_x*u_xxt - \
                     3*df[:, 2]*u_xt*u_xx - 3*df[:, 3]*u_x**2*u_xt - \
                     3*df[:, 3]*u_t*u_x*u_xx
            u_xttt = -df[:, 1]*u_xxtt - 2*df[:, 2]*u_xt**2 - \
                     2*df[:, 2]*u_t*u_xxt - 2*df[:, 2]*u_x*u_xtt - \
                     df[:, 2]*u_tt*u_xx - df[:, 3]*u_x**2*u_tt -\
                     df[:, 3]*u_t**2*u_xx - df[:, 4]*u_t**2*u_x**2 - \
                     4*df[:, 3]*u_t*u_x*u_xt
            u_tttt = -df[:, 1]*u_xttt - df[:, 4]*u_t**3*u_x - \
                     3*df[:, 2]*u_t*u_xtt - df[:, 2]*u_x*u_ttt - \
                     3*df[:, 2]*u_tt*u_xt - 3*df[:, 3]*u_t**2*u_xt - \
                     3*df[:, 3]*u_t*u_x*u_tt
            du_dt[:, 4] = u_tttt
        if N >= 5:
            u_xxxxx = du_dx[:, 5]
            u_xxxxt = -df[:, 5]*u_x**5 - df[:, 1]*u_xxxxx - \
                      15*df[:, 3]*u_x*u_xx**2 - 10*df[:, 4]*u_x**3*u_xx - \
                      5*df[:, 2]*u_x*u_xxxx - 10*df[:, 2]*u_xx*u_xxx - \
                      10*df[:, 3]*u_x**2*u_xxx
            u_xxxtt = -df[:, 1]*u_xxxxt - 3*df[:, 3]*u_t*u_xx**2 - \
                      df[:, 5]*u_t*u_x**4 - 4*df[:, 4]*u_x**3*u_xt - \
                      df[:, 2]*u_t*u_xxxx - 4*df[:, 2]*u_x*u_xxxt - \
                      4*df[:, 2]*u_xt*u_xxx - 6*df[:, 2]*u_xx*u_xxt - \
                      6*df[:, 3]*u_x**2*u_xxt - 4*df[:, 3]*u_t*u_x*u_xxx - \
                      12*df[:, 3]*u_x*u_xt*u_xx - 6*df[:, 4]*u_t*u_x**2*u_xx
            u_xxttt = -df[:, 1]*u_xxxtt - 6*df[:, 3]*u_x*u_xt**2 - \
                      df[:, 4]*u_x**3*u_tt - 2*df[:, 2]*u_t*u_xxxt - \
                      3*df[:, 2]*u_x*u_xxtt - df[:, 2]*u_tt*u_xxx - \
                      6*df[:, 2]*u_xt*u_xxt - 3*df[:, 2]*u_xx*u_xtt - \
                      df[:, 5]*u_t**2*u_x**3 - 3*df[:, 3]*u_x**2*u_xtt - \
                      df[:, 3]*u_t**2*u_xxx - 6*df[:, 3]*u_t*u_x*u_xxt - \
                      6*df[:, 3]*u_t*u_xt*u_xx - 3*df[:, 3]*u_x*u_tt*u_xx - \
                      6*df[:, 4]*u_t*u_x**2*u_xt - 3*df[:, 4]*u_t**2*u_x*u_xx
            u_xtttt = -df[:, 1]*u_xxttt - 6*df[:, 3]*u_t*u_xt**2 - \
                      df[:, 4]*u_t**3*u_xx - 3*df[:, 2]*u_t*u_xxtt - \
                      2*df[:, 2]*u_x*u_xttt - 3*df[:, 2]*u_tt*u_xxt - \
                      6*df[:, 2]*u_xt*u_xtt - df[:, 2]*u_xx*u_ttt - \
                      df[:, 5]*u_t**3*u_x**2 - df[:, 3]*u_x**2*u_ttt - \
                      3*df[:, 3]*u_t**2*u_xxt - 6*df[:, 3]*u_t*u_x*u_xtt - \
                      3*df[:, 3]*u_t*u_tt*u_xx - 6*df[:, 3]*u_x*u_tt*u_xt - \
                      3*df[:, 4]*u_t*u_x**2*u_tt - 6*df[:, 4]*u_t**2*u_x*u_xt
            u_ttttt = -df[:, 1]*u_xtttt - 3*df[:, 3]*u_x*u_tt**2 - \
                      df[:, 5]*u_t**4*u_x - 4*df[:, 4]*u_t**3*u_xt - \
                      4*df[:, 2]*u_t*u_xttt - df[:, 2]*u_x*u_tttt - \
                      6*df[:, 2]*u_tt*u_xtt - 4*df[:, 2]*u_xt*u_ttt - \
                      6*df[:, 3]*u_t**2*u_xtt - 4*df[:, 3]*u_t*u_x*u_ttt - \
                      12*df[:, 3]*u_t*u_tt*u_xt - 6*df[:, 4]*u_t**2*u_x*u_tt
            du_dt[:, 5] = u_ttttt
        if N >= 6:
            u_xxxxxx = du_dx[:, 6]
            u_xxxxxt = -15*df[:, 3]*u_xx**3 - 10*df[:, 2]*u_xxx**2 - \
                       df[:, 6]*u_x**6 - df[:, 1]*u_xxxxxx - \
                       20*df[:, 4]*u_x**3*u_xxx - 6*df[:, 2]*u_x*u_xxxxx - \
                       15*df[:, 2]*u_xx*u_xxxx - 15*df[:, 3]*u_x**2*u_xxxx - \
                       15*df[:, 5]*u_x**4*u_xx - 45*df[:, 4]*u_x**2*u_xx**2 - \
                       60*df[:, 3]*u_x*u_xx*u_xxx
            u_xxxxtt = -df[:, 1]*u_xxxxxt - df[:, 2]*u_t*u_xxxxx - \
                       5*df[:, 2]*u_x*u_xxxxt - 5*df[:, 2]*u_xt*u_xxxx - \
                       10*df[:, 2]*u_xx*u_xxxt - 10*df[:, 2]*u_xxt*u_xxx - \
                       10*df[:, 3]*u_x**2*u_xxxt - df[:, 6]*u_t*u_x**5 - \
                       15*df[:, 3]*u_xt*u_xx**2 - 5*df[:, 5]*u_x**4*u_xt - \
                       10*df[:, 4]*u_x**3*u_xxt - 10*df[:, 5]*u_t*u_x**3*u_xx -\
                       5*df[:, 3]*u_t*u_x*u_xxxx - 10*df[:, 3]*u_t*u_xx*u_xxx -\
                       20*df[:, 3]*u_x*u_xt*u_xxx - \
                       30*df[:, 3]*u_x*u_xx*u_xxt - \
                       10*df[:, 4]*u_t*u_x**2*u_xxx - \
                       30*df[:, 4]*u_x**2*u_xt*u_xx - \
                       15*df[:, 4]*u_t*u_x*u_xx**2
            u_xxxttt = -6*df[:, 2]*u_xxt**2 - df[:, 1]*u_xxxxtt - \
                       2*df[:, 2]*u_t*u_xxxxt - 4*df[:, 2]*u_x*u_xxxtt - \
                       df[:, 2]*u_tt*u_xxxx - 8*df[:, 2]*u_xt*u_xxxt - \
                       6*df[:, 2]*u_xx*u_xxtt - 4*df[:, 2]*u_xtt*u_xxx - \
                       6*df[:, 3]*u_x**2*u_xxtt - df[:, 3]*u_t**2*u_xxxx - \
                       3*df[:, 3]*u_tt*u_xx**2 - 12*df[:, 3]*u_xt**2*u_xx - \
                       df[:, 5]*u_x**4*u_tt - 3*df[:, 4]*u_t**2*u_xx**2 - \
                       12*df[:, 4]*u_x**2*u_xt**2 - df[:, 6]*u_t**2*u_x**4 - \
                       4*df[:, 4]*u_x**3*u_xtt - 8*df[:, 5]*u_t*u_x**3*u_xt - \
                       8*df[:, 3]*u_t*u_x*u_xxxt - 8*df[:, 3]*u_t*u_xt*u_xxx - \
                       12*df[:, 3]*u_t*u_xx*u_xxt - 4*df[:, 3]*u_x*u_tt*u_xxx -\
                       24*df[:, 3]*u_x*u_xt*u_xxt - \
                       12*df[:, 3]*u_x*u_xx*u_xtt - \
                       12*df[:, 4]*u_t*u_x**2*u_xxt - \
                       6*df[:, 4]*u_x**2*u_tt*u_xx - \
                       4*df[:, 4]*u_t**2*u_x*u_xxx - \
                       6*df[:, 5]*u_t**2*u_x**2*u_xx - \
                       24*df[:, 4]*u_t*u_x*u_xt*u_xx
            u_xxtttt = -6*df[:, 3]*u_xt**3 - df[:, 1]*u_xxxttt - \
                       3*df[:, 3]*u_x**2*u_xttt - 3*df[:, 3]*u_t**2*u_xxxt - \
                       df[:, 6]*u_t**3*u_x**3 - df[:, 4]*u_x**3*u_ttt - \
                       df[:, 4]*u_t**3*u_xxx - 3*df[:, 2]*u_t*u_xxxtt - \
                       3*df[:, 2]*u_x*u_xxttt - 3*df[:, 2]*u_tt*u_xxxt - \
                       9*df[:, 2]*u_xt*u_xxtt - 3*df[:, 2]*u_xx*u_xttt - \
                       df[:, 2]*u_ttt*u_xxx - 9*df[:, 2]*u_xtt*u_xxt - \
                       3*df[:, 5]*u_t**3*u_x*u_xx - 9*df[:, 3]*u_t*u_x*u_xxtt -\
                       3*df[:, 3]*u_t*u_tt*u_xxx - 18*df[:, 3]*u_t*u_xt*u_xxt -\
                       9*df[:, 3]*u_t*u_xx*u_xtt - 9*df[:, 3]*u_x*u_tt*u_xxt - \
                       18*df[:, 3]*u_x*u_xt*u_xtt - 3*df[:, 3]*u_x*u_xx*u_ttt -\
                       9*df[:, 3]*u_tt*u_xt*u_xx - \
                       9*df[:, 4]*u_t*u_x**2*u_xtt - \
                       9*df[:, 4]*u_x**2*u_tt*u_xt - \
                       9*df[:, 4]*u_t**2*u_x*u_xxt - \
                       9*df[:, 4]*u_t**2*u_xt*u_xx - \
                       9*df[:, 5]*u_t**2*u_x**2*u_xt - \
                       18*df[:, 4]*u_t*u_x*u_xt**2 - \
                       3*df[:, 5]*u_t*u_x**3*u_tt - \
                       9*df[:, 4]*u_t*u_x*u_tt*u_xx
            u_xttttt = -6*df[:, 2]*u_xtt**2 - df[:, 1]*u_xxtttt - \
                       df[:, 3]*u_x**2*u_tttt - 6*df[:, 3]*u_t**2*u_xxtt - \
                       12*df[:, 3]*u_tt*u_xt**2 - 3*df[:, 3]*u_tt**2*u_xx - \
                       df[:, 5]*u_t**4*u_xx - 12*df[:, 4]*u_t**2*u_xt**2 - \
                       3*df[:, 4]*u_x**2*u_tt**2 - df[:, 6]*u_t**4*u_x**2 - \
                       4*df[:, 4]*u_t**3*u_xxt - 4*df[:, 2]*u_t*u_xxttt - \
                       2*df[:, 2]*u_x*u_xtttt - 6*df[:, 2]*u_tt*u_xxtt - \
                       8*df[:, 2]*u_xt*u_xttt - df[:, 2]*u_xx*u_tttt - \
                       4*df[:, 2]*u_ttt*u_xxt - 8*df[:, 5]*u_t**3*u_x*u_xt - \
                       8*df[:, 3]*u_t*u_x*u_xttt - 12*df[:, 3]*u_t*u_tt*u_xxt -\
                       24*df[:, 3]*u_t*u_xt*u_xtt - 4*df[:, 3]*u_t*u_xx*u_ttt -\
                       12*df[:, 3]*u_x*u_tt*u_xtt - 8*df[:, 3]*u_x*u_xt*u_ttt -\
                       4*df[:, 4]*u_t*u_x**2*u_ttt - \
                       12*df[:, 4]*u_t**2*u_x*u_xtt - \
                       6*df[:, 4]*u_t**2*u_tt*u_xx - \
                       6*df[:, 5]*u_t**2*u_x**2*u_tt - \
                       24*df[:, 4]*u_t*u_x*u_tt*u_xt
            u_tttttt = -df[:, 1]*u_xttttt - 10*df[:, 3]*u_t**2*u_xttt - \
                       df[:, 6]*u_t**5*u_x - 15*df[:, 3]*u_tt**2*u_xt - \
                       5*df[:, 5]*u_t**4*u_xt - 10*df[:, 4]*u_t**3*u_xtt - \
                       5*df[:, 2]*u_t*u_xtttt - df[:, 2]*u_x*u_ttttt - \
                       10*df[:, 2]*u_tt*u_xttt - 5*df[:, 2]*u_xt*u_tttt - \
                       10*df[:, 2]*u_ttt*u_xtt - 10*df[:, 4]*u_t**2*u_x*u_ttt -\
                       30*df[:, 4]*u_t**2*u_tt*u_xt - \
                       15*df[:, 4]*u_t*u_x*u_tt**2 - \
                       10*df[:, 5]*u_t**3*u_x*u_tt  - \
                       5*df[:, 3]*u_t*u_x*u_tttt - 30*df[:, 3]*u_t*u_tt*u_xtt -\
                       20*df[:, 3]*u_t*u_xt*u_ttt - 10*df[:, 3]*u_x*u_tt*u_ttt
            du_dt[:, 6] = u_tttttt
        if N not in range(1, 7):
            raise NotImplementedError("Cauchy Kovalevskaya is only " + \
                                      "implemented for N <=6.")
        return du_dt

class Linear(Equation):

    def __init__(self, a=1.0):
        self.a = a
        super().__init__("linear advection equation")

    def flux(self, u):
        return np.array([self.a*u])

    def flux_derivative(self, u, k=1):
        if k == 0:
            return self.flux(u)
        elif k == 1:
            return np.array([self.a])
        elif k >= 2:
            return np.array([0.0])
        else:
            raise NotImplementedError("k has to be a non-negative integer")

    def godunov_state(self, u_L, u_R):
        if self.a >= 0:
            return u_L
        else:
            return u_R

    def cauchy_kovalevskaya(self, du_dx):
        N = du_dx.shape[1] - 1
        du_dt = np.empty(du_dx.shape)
        for k in range(N + 1):
            du_dt[:, k] = (-self.a)**k*du_dx[:, k]
        return du_dt

class Burgers(Equation):

    def __init__(self):
        super().__init__("Burgers equation")

    def flux(self, u):
        return np.array([0.5*u**2])

    def flux_derivative(self, u, k=1):
        if k == 0:
            return self.flux(u)
        elif k == 1:
            return np.array([u])
        elif k == 2:
            return np.array([1.0])
        elif k >= 3:
            return np.array([0.0])
        else:
            raise NotImplementedError("k has to be a non-negative integer")

    def godunov_state(self, u_L, u_R):
        if isinstance(u_L, np.ndarray):
            u_L = u_L[0]
        if isinstance(u_R, np.ndarray):
            u_R = u_R[0]
        if u_L > u_R:
            # shock
            s = 0.5*(u_R + u_L)
            if 0 <= s:
                return u_L
            else:
                return u_R
        else:
            # rarefaction wave
            if 0 <= u_L:
                return u_L
            elif u_L < 0 and 0 < u_R:
                return 0
            else:
                return u_R

class Traffic(Equation):

    def __init__(self, rho_max=1.0, v_max=1.0):
        if rho_max <= 0.0 or v_max <= 0.0:
            raise NotImplementedError("rho_max and v_max should be bigger " + \
                                      "than 0 in traffic flow equation.")
        self.rho_max = rho_max
        self.v_max = v_max
        super().__init__("traffic flow equation")
    
    def flux(self, u):
        return np.array([u*(1 - u/self.rho_max)*self.v_max])

    def flux_derivative(self, u, k=1):
        if k == 0:
            return self.flux(u)
        elif k == 1:
            return np.array([(1 - 2*u/self.rho_max)*self.v_max])
        elif k == 2:
            return np.array([-2/self.rho_max*self.v_max])
        elif k >= 3:
            return np.array([0.0])
        else:
            raise NotImplementedError("k has to be a non-negative integer")

    def godunov_state(self, u_L, u_R):
        if isinstance(u_L, np.ndarray):
            u_L = u_L[0]
        if isinstance(u_R, np.ndarray):
            u_R = u_R[0]
        if u_L < u_R:
            # shock
            s = self.v_max*(1 - (u_L + u_R)/self.rho_max)
            if 0 <= s:
                return u_L
            else:
                return u_R
        else:
            # rarefaction wave
            if 1/2*self.rho_max >= u_L:
                return u_L
            elif u_L > 1/2*self.rho_max and 1/2*self.rho_max > u_R:
                return 1/2*self.rho_max
            else:
                return u_R

class Cubic(Equation):

    def __init__(self):
        super().__init__("cubic equation")

    def flux(self, u):
        return np.array([u**3/3])

    def flux_derivative(self, u, k=1):
        if k == 0:
            return self.flux(u)
        elif k == 1:
            return np.array([u**2])
        elif k == 2:
            return np.array([2*u])
        elif k == 3:
            return np.array([2.0])
        elif k >= 4:
            return np.array([0.0])
        else:
            raise NotImplementedError("k has to be a non-negative integer")

    def godunov_state(self, u_L, u_R):
        if isinstance(u_L, np.ndarray):
            u_L = u_L[0]
        if isinstance(u_R, np.ndarray):
            u_R = u_R[0]
        if u_L < 0 and u_R < 0:
            # concave case
            if u_L < u_R:
                # shock
                s = 1/3*(u_L**2 + u_L*u_R + u_R**2)
                if 0 <= s:
                    return u_L
                else:
                    return u_R
            else:
                # rarefaction wave
                return u_L
        elif u_L > 0 and u_R > 0:
            # convex case
            if u_L > u_R:
                # shock
                s = 1/3*(u_L**2 + u_L*u_R + u_R**2)
                if 0 <= s:
                    return u_L
                else:
                    return u_R
            else:
                # rarefaction wave
                return u_L
        else:
            raise ValueError("u_L and u_R should have the same sign. " + \
                             "Otherwise the flux is neither convex nor " + \
                             "concave.")
