# coding: utf-8
import numpy as np
from scipy import interpolate
from devito.interfaces import DenseData
from devito.dimension import time, t

class IGrid:
    """
    Class to setup a physical model

    :param origin: Origin of the model in m as a Tuple
    :param spacing:grid size in m as a Tuple
    :param vp: Velocity in km/s
    :param rho: Density in kg/cm^3 (rho=1 for water)
    :param epsilon: Thomsen epsilon parameter (0<epsilon<1)
    :param delta: Thomsen delta parameter (0<delta<1), delta<epsilon
    :param: theta: Tilt angle in radian
    :param phi : Asymuth angle in radian
    """
    def __init__(self, origin, spacing, dimensions, vp, rho=None, epsilon=None,
                 delta=None, theta=None, phi=None, nbpml=40):
        self.vp = vp
        self.origin = origin
        self.spacing = spacing
        self.nbpml = nbpml
        self.shape = dimensions
        self.dtype = np.float32
        # Create square slowness of the wave as symbol `m`
        if isinstance(vp, np.ndarray):
            self.m = DenseData(name="m", shape=self.shape_domain, dtype=self.dtype)
        else:
            self.m = 1/vp**2
        # Create dampening field as symbol `damp`
        self.damp = DenseData(name="damp", shape=self.shape_domain,
                              dtype=self.dtype)
        # Additional parameter fields for TTI operators
        if rho is not None:
            if isinstance(rho, np.ndarray):
                self.rho = DenseData(name="rho", shape=self.shape_domain,
                                     dtype=self.dtype)
            else:
                self.rho = rho
        else:
            self.rho = 1

    @property
    def shape_domain(self):
        """Computational shape of the model domain, with PML layers"""
        return tuple(d + 2*self.nbpml for d in self.shape)

    @property
    def critical_dt(self):
        # limit for infinite stencil of √(a1/a2) where a1 is the
        #  sum of absolute values of the time discretisation
        # and a2 is the sum of the absolute values of the space discretisation
        #
        # example, 2nd order in time and space in 2D
        # a1 = 1 + 2 + 1 = 4
        # a2 = 2*(1+2+1)  = 8
        # coeff = √(1/2) = 0.7
        # example, 2nd order in time and space in 3D
        # a1 = 1 + 2 + 1 = 4
        # a2 = 3*(1+2+1)  = 12
        # coeff = √(1/3) = 0.57

        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        return coeff * np.min(self.spacing) / (np.max(self.vp))

    def get_spacing(self):
        return self.spacing


    def set_vp(self, vp):
        self.vp = vp

    def set_origin(self, shift):
        norig = len(self.origin)
        aux = []

        for i in range(0, norig):
            aux.append(self.origin[i] - shift * self.spacing[i])

        self.origin = aux

    def get_origin(self):
        return self.origin

    def padm(self):
        return self.pad(self.vp**(-2))

    def pad(self, m):
        pad_list = []
        for dim_index in range(len(self.vp.shape)):
            pad_list.append((self.nbpml, self.nbpml))
        return np.pad(m, pad_list, 'edge')

class IShot:
    def get_data(self):
        """ List of ISource objects, of size ntraces
        """
        return self._shots

    def set_source(self, time_serie, dt, location):
        """ Depreciated"""
        self.source_sign = time_serie
        self.source_coords = location
        self.sample_interval = dt

    def set_receiver_pos(self, pos):
        """ Position of receivers as an
         (nrec, 3) array"""
        self.receiver_coords = pos

    def set_shape(self, nt, nrec):
        """ Shape of the shot record
        (nt, nrec)"""
        self.shape = (nt, nrec)
        self.nt = nt

    def set_traces(self, traces):
        """ Add traces data  """
        self.traces = traces

    def set_time_axis(self, dt, tn):
        """ Define the shot record time axis
        with sampling interval and last time"""
        self.sample_interval = dt
        self.end_time = tn

    def get_source(self, ti=None):
        """ Depreciated"""
        if ti is None:
            return self.source_sign

        return self.source_sign[ti]

    def get_nrec(self):
        """ List of ISource objects, of size ntraces
                """
        ntraces, nsamples = self.traces.shape

        return ntraces

    def reinterpolate(self, dt, order=3):
        """ Reinterpolate data onto a new time axis """
        if np.isclose(dt, self.sample_interval):
            return

        nsamples, ntraces = self.shape

        oldt = np.arange(0, self.end_time + self.sample_interval,
                         self.sample_interval)
        newt = np.arange(0, self.end_time + dt, dt)

        new_nsamples = len(newt)
        new_traces = np.zeros((new_nsamples, ntraces), np.float32)

        if hasattr(self, 'traces'):
            for i in range(ntraces):
                tck = interpolate.splrep(oldt, self.traces[:, i], s=0, k=order)
                new_traces[:, i] = interpolate.splev(newt, tck)

        self.traces = new_traces
        self.sample_interval = dt
        self.nsamples = new_nsamples
        self.shape = new_traces.shape

    def reinterpolateD(self, datain, dtin, dtout, order=3):
        """ Reinterpolate an input array onto a new time axis"""
        if np.isclose(dtin, dtout):
            return datain

        if len(datain.shape) == 1:
            nsamples = datain.shape
            ntraces = 1
        else:
            nsamples, ntraces = datain.shape

        ntin = int(self.end_time/dtin + 1)
        ntout = int(self.end_time/dtout + 1)
        oldt = np.linspace(0, self.end_time, ntin)
        newt = np.linspace(0, self.end_time, ntout)

        new_nsamples = len(newt)
        new_traces = np.zeros((new_nsamples, ntraces), np.float32)

        if len(datain.shape) == 1:
            tck = interpolate.splrep(oldt, datain, s=0, k=order)
            new_traces = np.float32(interpolate.splev(newt, tck))
        else:
            for i in range(ntraces):
                tck = interpolate.splrep(oldt, datain[:, i], s=0, k=order)
                new_traces[:, i] = interpolate.splev(newt, tck)
        return new_traces

    def __str__(self):
        return "Source: "+str(self.source_coords)+", Receiver:"+str(self.receiver_coords)
