import numpy as np
import warnings


class TimeMesh:

    def __init__(self, t_0, t_end, N=None, dt=None):
        if N is None and dt is None:
            raise ValueError("Either N or dt has to be a non-None value." +
                             " If you use a StepsizeCallback provide a dummy" +
                             " dt, that will be overwritten.")
        if N is not None and dt is not None:
            warnings.warn("Only one of N and dt should be a non-None value." +
                          " Ignoring N and take only take dt")
        if dt is None:
            self.dt = (t_end - t_0) / N
        else:
            self.dt = dt
        self._original_dt = self.dt
        self.t_end = t_end
        self.t = np.array([t_0])

    @property
    def t_0(self):
        return self.t[0]

    def update(self, dt=None):
        time = self.t[-1]
        if dt is None:
            dt = self.dt
        if time + dt > self.t_end:
            self.dt = self.t_end - time
        else:
            self.dt = dt
        self.t = np.append(self.t, time + self.dt)

    def isfinished(self):
        if self.t[-1] >= self.t_end:
            return True
        return False

    def reset(self):
        self.t = np.array([self.t_0])
        self.dt = self._original_dt


class SpatialMesh:

    def __init__(self, xmin, xmax, Nx):
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        self.dx = (xmax - xmin)/Nx
        self.x = np.linspace(xmin + self.dx/2, xmax - self.dx/2, Nx)

class Mesh:

    def __init__(self, xmin, xmax, t_0, t_end, Nx, N=None, dt=None):
        self.timemesh = TimeMesh(t_0, t_end, N, dt)
        self.spatialmesh = SpatialMesh(xmin, xmax, Nx)

    def update(self, dt=None):
        self.timemesh.update(dt)

    def isfinished(self):
        return self.timemesh.isfinished()

    def reset(self):
        self.timemesh.reset()
