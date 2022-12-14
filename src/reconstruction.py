import numpy as np

from .util import boundary_condition


class WENOReconstruction:

    def __init__(self, N, bc="periodic", eps=1e-13, r=8, lambdas=[1, 1e5, 1]):
        self.N = N
        self.bc = bc
        self.eps = eps
        self.r = r
        self.lambdas = lambdas
        if N == 1:
            w_hat = np.array([[-1/2, 1/2],
                                  [-1/2, 1/2]])

            def minmod(a, b):
                return (np.sign(a) == np.sign(b)) * \
                       np.sign(a) * np.min([np.abs(a), np.abs(b)])
            self.limiter = minmod
        elif N == 2:
            w_hat = np.array([[1/4, -1.0, 3/4],
                              [1/12, -1/6, 1/12],
                                  [-1/4, 0.0, 1/4],
                                  [1/12, -1/6, 1/12],
                                      [-3/4, 1.0, -1/4],
                                      [1/12, -1/6, 1/12]])
            beta_coeff = np.array([156, 4])

            def a_comb(a):
                return np.array([a[:, :, 2]**2, a[:, :, 1]**2]).T
        elif N == 3:
            w_hat = np.array([[-19/120, 87/120, -177/120, 109/120],
                              [-1/12, 1/3, -5/12, 1/6],
                              [-1/120, 1/40, -1/40, 1/120],
                                   [11/120, -63/120, 33/120, 19/120],
                                   [0.0, 1/12, -1/6, 1/12],
                                   [-1/120, 1/40, -1/40, 1/120],
                                       [-19/120, -33/120, 63/120, -11/120],
                                       [1/12, -1/6, 1/12, 0.0],
                                       [-1/120, 1/40, -1/40, 1/120],
                                           [-109/120, 177/120, -87/120,
                                            19/120],
                                           [1/6, -5/12, 1/3, -1/12],
                                           [-1/120, 1/40, -1/40, 1/120]])
            beta_coeff = np.array([1224, 8, 156, 4])

            def a_comb(a):
                return np.array([a[:, :, 3]**2, a[:, :, 3]*a[:, :, 1],
                                 a[:, :, 2]**2, a[:, :, 1]**2]).T
        elif N == 4:
            w_hat = np.array([[27/240, -146/240, 336/240, -462/240, 245/240],
                              [25/336, -128/336, 262/336, -240/336, 81/336],
                              [3/240, -14/240, 24/240, -18/240, 5/240],
                              [1/1680, -4/1680, 6/1680, -4/1680, 1/1680],
                                  [-11/240, 66/240, -192/240, 110/240, 27/240],
                                  [-3/336, 12/336, 10/336, -44/336, 25/336],
                                  [1/240, -6/240, 12/240, -10/240, 3/240],
                                  [1/1680, -4/1680, 6/1680, -4/1680, 1/1680],
                                      [11/240, -82/240, 0.0, 82/240, -11/240],
                                      [-3/336, 40/336, -74/336, 40/336,
                                       -3/336],
                                      [-1/240, 2/240, 0.0, -2/240, 1/240],
                                      [1/1680, -4/1680, 6/1680, -4/1680,
                                       1/1680],
                              [-27/240, -110/240, 192/240, -66/240, 11/240],
                              [25/336, -44/336, 10/336, 12/336, -3/336],
                              [-3/240, 10/240, -12/240, 6/240, -1/336],
                              [1/1680, -4/1680, 6/1680, -4/1680, 1/1680],
                                  [-245/240, 462/240, -336/240, 146/240,
                                   -27/240],
                                  [81/336, -240/336, 262/336, -128/336,
                                   25/336],
                                  [-5/240, 18/240, -24/240, 14/240, -3/240],
                                  [1/1680, -4/1680, 6/1680, -4/1680, 1/1680]])
            beta_coeff = 4*np.array([765790, 246, 3906, 2, 39, 1])

            def a_comb(a):
                return np.array([a[:, :, 4]**2, a[:, :, 4]*a[:, :, 2],
                                 a[:, :, 3]**2, a[:, :, 3]*a[:, :, 1],
                                 a[:, :, 2]**2, a[:, :, 1]**2]).T
        elif N == 5:
            w_hat = np.array([[-863/10080, 5449/10080, -14762/10080,
                               22742/10080, -23719/10080, 11153/10080],
                              [-22/336, 135/336, -348/336, 482/336, -350/336,
                               103/336],
                              [-31/2160, 182/2160, -436/2160, 526/2160,
                               -317/2160, 76/2160],
                              [-2/1680, 11/1680, -24/1680, 26/1680, -14/1680,
                               3/1680],
                              [-1/30240, 5/30240, -10/30240, 10/30240,
                               -5/30240, 1/30240],
                                  [271/10080, -1817/10080, 5482/10080,
                                   -10774/10080, 5975/10080, 863/10080],
                                  [3/336, -18/336, 42/336, -20/336, -29/336,
                                   22/336],
                                  [-4/2160, 29/2160, -94/2160, 148/2160,
                                   -110/2160, 31/2160],
                                  [-1/1680, 6/1680, -14/1680, 16/1680, -9/1680,
                                   2/1680],
                                  [-1/30240, 5/30240, -10/30240, 10/30240,
                                   -5/30240, 1/30240],
                                      [-191/10080, 1417/10080, -5354/10080,
                                       1910/10080, 2489/10080, -271/10080],
                                      [0.0, -3/336, 40/336, -74/336, 40/336,
                                       -3/336],
                                      [5/2160, -34/2160, 68/2160, -50/2160,
                                       7/2160, 4/2160],
                                      [0.0, 1/1680, -4/1680, 6/1680, -4/1680,
                                       1/1680],
                                      [-1/30240, 5/30240, -10/30240,
                                       10/30240, -5/30240, 1/30240],
                                          [271/10080, -2489/10080, -1910/10080,
                                           5354/10080, -1417/10080, 191/10080],
                                          [-3/336, 40/336, -74/336, 40/336,
                                           -3/336, 0.0],
                                          [-4/2160, -7/2160, 50/2160, -68/2160,
                                           34/2160, -5/2160],
                                          [1/1680, -4/1680, 6/1680, -4/1680,
                                           1/1680, 0.0],
                                          [-1/30240, 5/30240, -10/30240,
                                           10/30240, -5/30240, 1/30240],
                              [-863/10080, -5975/10080, 10774/10080,
                               -5482/10080, 1817/10080, -217/10080],
                              [22/336, -29/336, -20/336, 42/336, -18/336,
                               3/336],
                              [-31/2160, 110/2160, -148/2160, 94/2160,
                               -29/2160, 4/2160],
                              [2/1680, -9/1680, 16/1680, -14/1680, 6/1680,
                               -1/1680],
                              [-1/30240, 5/30240, -10/30240, 10/30240,
                               -5/30240, 1/30240],
                                  [-11153/10080, 23719/10080, -22742/10080,
                                   14762/10080, -5549/10080, 863/10080],
                                  [103/336, -350/336, 482/336, -348/336,
                                   135/336, -22/336],
                                  [-76/2160, 317/2160, -526/2160, 436/2160,
                                   -182/2160, 31/2160],
                                  [3/1680, -14/1680, 26/1680, -24/1680,
                                   11/1680, -2/1680],
                                  [-1/30240, 5/30240, -10/30240, 10/30240,
                                   -5/30240, 1/30240]])
            beta_coeff = 4*np.array([248164155, 52092, 2, 765790, 246, 3906, 2,
                                     39, 1])

            def a_comb(a):
                return np.array([a[:, :, 5]**2, a[:, :, 3]*a[:, :, 5],
                                 a[:, :, 1]*a[:, :, 5], a[:, :, 4]**2,
                                 a[:, :, 2]*a[:, :, 4], a[:, :, 3]**2,
                                 a[:, :, 1]*a[:, :, 3], a[:, :, 2]**2,
                                 a[:, :, 1]**2]).T
        elif N == 6:
            w_hat = np.array([[1375/20160, -9976/20160, 31523/20160,
                               -57014/20160, 66109/20160, -55688/20160,
                               23681/20160],
                              [3499/60480, -24954/60480, 76785/60480,
                               -132620/60480, 139245/60480, -83994/60480,
                               22039/60480],
                              [65/4320, -452/4320, 1339/4320, -2172/4320,
                               2027/4320, -1024/4320, 217/4320],
                              [185/110880, -1242/110880, 3501/110880,
                               -5284/110880, 4491/110880, -2034/110880,
                               383/110880],
                              [5/60480, -32/60480, 85/60480, -120/60480,
                               95/60480, -40/60480, 7/60480],
                              [1/665280, -6/665280, 15/665280, -20/665280,
                               15/665280, -6/665280, 1/665280],
                                  [-351/20160, 2648/20160, -8899/20160,
                                   17984/20160, -26813/20160, 14056/20160,
                                   1375/20160],
                                  [-461/60480, 3306/60480, -10155/60480,
                                   16780/60480, -10515/60480, 2454/60480,
                                   3499/60480],
                                  [3/4320, -26/4320, 103/4320, -248/4320,
                                   341/4320, -238/4320, 65/4320],
                                  [53/110880, -384/110880, 1191/110880,
                                   -1984/110880, 1851/110880, -912/110880,
                                   185/110880],
                                  [3/60480, -20/60480, 55/60480, -80/60480,
                                   65/60480, -28/60480, 5/60480],
                                  [1/665280, -6/665280, 15/665280, -20/665280,
                                   15/665280, -6/665280, 1/665280],
                                      [191/20160, -1528/20160, 5699/20160,
                                       -14528/20160, 6685/20160, 3832/20160,
                                       -351/20160],
                                      [79/60480, -474/60480, 645/60480,
                                       5620/60480, -12135/60480, 6726/60480,
                                       -461/60480],
                                      [-5/4320, 40/4320, -143/4320, 236/4320,
                                       -175/4320, 44/4320, 3/4320],
                                      [-13/110880, 78/110880, -129/110880,
                                       -4/110880, 201/110880, -186/110880,
                                       53/110880],
                                      [1/60480, -8/60480, 25/60480, -40/60480,
                                       35/60480, -16/60480, 3/60480],
                                      [1/665280, -6/665280, 15/665280,
                                       -20/665280, 15/665280, -6/665280,
                                       1/665280],
                              [-191/20160, 1688/20160, -7843/20160, 0.0,
                               7843/20160, -1688/20160, 191/20160],
                              [79/60480, -1014/60480, 8385/60480, -14900/60480,
                               8385/60480, -1014/60480, 79/60480],
                              [5/4320, -38/4320, 61/4320, 0.0, -61/4320,
                               38/4320, -5/4320],
                              [-13/110880, 144/110880, -459/110880,
                               656/110880, -459/110880, 144/110880,
                               -13/110880],
                              [-1/60480, 4/60480, -5/60480, 0.0, 5/60480,
                               -4/60480, 1/60480],
                              [1/665280, -6/665280, 15/665280, -20/665280,
                               15/665280, -6/665280, 1/665280],
                                  [351/20160, -3832/20160, -6685/20160,
                                   14528/20160, -5699/20160, 1528/20160,
                                   -191/20160],
                                  [-461/60480, 6726/60480, -12135/60480,
                                   5620/60480, 645/60480, -474/60480,
                                   79/60480],
                                  [-3/4320, -44/4320, 175/4320, -236/4320,
                                   143/4320, -40/4320, 5/4320],
                                  [53/110880, -186/110880, 201/110880,
                                   -4/110880, -129/110880, 78/110880,
                                   -13/110880],
                                  [-3/60480, 16/60480, -35/60480, 40/60480,
                                   -25/60480, 8/60480, 1/60480],
                                  [1/665280, -6/665280, 15/665280, -20/665280,
                                   15/665280, -6/665280, 1/665280],
                                      [-1375/20160, -14056/20160, 26813/20160,
                                       -17984/20160, 8899/20160, -2648/20160,
                                       351/20160],
                                      [3499/60480, -2454/60480, -10515/60480,
                                       16780/60480, -10155/60480, 3306/60480,
                                       -461/60480],
                                      [-65/4320, 238/4320, -341/4320, 248/4320,
                                       -103/4320, 26/4320, -3/4320],
                                      [185/110880, -912/110880, 1851/110880,
                                       -1984/110880, 1191/110880, -384/110880,
                                       53/110880],
                                      [-5/60480, 28/60480, -65/60480, 80/60480,
                                       -55/60480, 20/60480, -3/60480],
                                      [1/665280, -6/665280, 15/665280,
                                       -20/665280, 15/665280, -6/665280,
                                       1/665280],
                              [-23681/20160, 55688/20160, -66109/20160,
                               57024/20160, -31523/20160, 9976/20160,
                               -1375/20160],
                              [22039/60480, -83994/60480, 139245/60480,
                               -132620/60480, 76785/60480, -24954/60480,
                               3499/60480],
                              [-217/4320, 1024/4320, -2027/4320, 2172/4320,
                               -1339/4320, 452/4320, -65/4320],
                              [383/110880, -2034/110880, 4491/110880,
                               -5284/110880, 3501/110880, -1242/110880,
                               185/110880],
                              [-7/60480, 40/60480, -95/60480, 120/60480,
                               -85/60480, 32/60480, -5/60480],
                              [1/665280, -6/665280, 15/665280, -20/665280,
                               15/665280, -6/665280, 1/665280]])
            beta_coeff = 4*np.array([120128261421, 17576180, 510, 248164155,
                                     52092, 2, 765790, 246, 3906, 2, 39, 1])

            def a_comb(a):
                return np.array([a[:, :, 6]**2, a[:, :, 6]*a[:, :, 4],
                                 a[:, :, 6]*a[:, :, 2], a[:, :, 5]**2,
                                 a[:, :, 5]*a[:, :, 3], a[:, :, 5]*a[:, :, 1],
                                 a[:, :, 4]**2, a[:, :, 4]*a[:, :, 2],
                                 a[:, :, 3]**2, a[:, :, 3]*a[:, :, 1],
                                 a[:, :, 2]**2, a[:, :, 1]**2]).T
        else:
            raise NotImplementedError("WENO reconstruction is only " +
                                      "implemented up to an order 6, but " +
                                      "got {}".format(N))
        self.w_hat = w_hat
        if N > 1:
            self.beta_coeff = beta_coeff
            self.a_comb = a_comb

    def reconstruct(self, u):
        N = self.N
        Nx = u.shape[1]
        w = np.empty((u.shape[0], N + 1, Nx))
        for j in range(Nx):
            # get all values of u that appear in at least one stencil
            u_stencil = np.empty((u.shape[0], 2*N + 1))
            for i in range(j - N, j + N + 1):
                ii = boundary_condition(i, Nx, self.bc)
                u_stencil[:, i + N - j] = u[:, ii]

            a = np.empty((u.shape[0], N + 1, N + 1))
            # for each of the N + 1 stencils
            for i in range(N + 1):
                # for each value in the stencil
                a[:, i, 0] = u_stencil[:, N]
                for k in range(N):
                    a[:, i, k + 1] = np.sum(self.w_hat[k + i*N] *
                                            u_stencil[:, i:i + N + 1],
                                            axis=1)
            if N == 1:
                a0 = u_stencil[:, N]
                a1 = self.limiter(a[:, 0, 1], a[:, 1, 1])
                w[:, 0, j] = a0
                w[:, 1, j] = a1
            else:
                # beta has shape (u.shape[0], N + 1)
                beta = np.sum(self.beta_coeff*self.a_comb(a), axis=2).T
                omega = np.empty((u.shape[0], N + 1))
                for i in range(N + 1):
                    if i < np.floor(N/2):
                        # left stencil
                        lambda_ = self.lambdas[0]
                    elif i > np.ceil(N/2):
                        # right stencil
                        lambda_ = self.lambdas[2]
                    else:
                        # center (one stencil if N odd, two if N is even)
                        lambda_ = self.lambdas[1]
                    omega[:, i] = lambda_/(self.eps + beta[:, i])**self.r
                # normalize
                omegasum = np.sum(omega, axis=1)
                omega /= omegasum
                for i in range(N + 1):
                    w[:, i, j] = np.sum(omega*a[:, :, i], axis=1)
        return w
