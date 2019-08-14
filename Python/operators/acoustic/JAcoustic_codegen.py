# coding: utf-8
from __future__ import print_function

import numpy as np

from devito.logger import set_log_level
from devito import Dimension, time, t, DenseData, TimeData
from Jsource_type import PointSource, Receiver
from Jfwi_operators import *

class JAcoustic_cg(object):
    """
    Class to setup the problem for the Acoustic Wave.

    Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, nbpml=40, t_order=2, s_order=8,
                 tsave=4.0):
        set_log_level('ERROR')
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.source = source
        self.dtype = np.float32
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        self.tsave = tsave

        if len(self.model.shape) == 2 and self.source.receiver_coords.shape[1] == 3:
            self.source.receiver_coords = np.delete(self.source.receiver_coords, 1, 1)
        if len(self.model.shape) == 2 and self.data.receiver_coords.shape[1] == 3:
            self.data.receiver_coords = np.delete(self.data.receiver_coords, 1, 1)


    def Forward(self, save=False, dse='advanced', 
                dle='advanced', q=False, free_surface=False):

        nt, nrec = self.data.shape
        nsrc = self.source.shape[1]
        ndim = len(self.model.shape)
        h = self.model.get_spacing()
        dtype = self.model.dtype
        nbpml = self.model.nbpml
        # Create source symbol
        src = PointSource(name='src', ntime=self.source.nt,
                          coordinates=self.source.receiver_coords)
        # Create receiver symbol
        rec = Receiver(name='rec', ntime=self.data.nt,
                       coordinates=self.data.receiver_coords)
        # Create the forward wavefield
        u = TimeData(name="u", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order, save=False,
                     dtype=self.model.dtype)
                     
        if q:
            qfull = TimeData(name="qfull", shape=self.model.shape_domain, time_dim=nt,
                             time_order=2, space_order=self.s_order, save=True,
                             dtype=self.model.dtype)
        else:
            qfull = 0
        # Execute operator and return wavefield and receiver data
        fw = ForwardOperator(self.model, u, src, rec, self.data, qfull,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, dse=dse, dle=dle, tsave=self.tsave,
                             free_surface=free_surface)

        return fw


    def Adjoint(self, cache_blocking=None,
                auto_tuning=False, profile=False, save=False,
                dse='advanced', dle='advanced', compiler=None,
                free_surface=False):
        """Adjoint modelling of a shot record.
        """
        nt, nrec = self.data.shape
        nsrc = self.source.shape[1]
        ndim = len(self.model.shape)
        h = self.model.get_spacing()
        dtype = self.model.dtype
        nbpml = self.model.nbpml

        # Create source symbol
        srca = PointSource(name='srca', ntime=self.source.nt,
                           coordinates=self.source.receiver_coords)

        # Create receiver symbol
        rec = Receiver(name='rec', ntime=self.data.nt,
                       coordinates=self.data.receiver_coords)
        # Create the forward wavefield
        v = TimeData(name="v", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order, save=save,
                     dtype=self.model.dtype)

        # Execute operator and return wavefield and receiver data
        adj = AdjointOperator(self.model, v, srca, rec, self.data,
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking, dse=dse,
                              dle=dle, free_surface=free_surface)
        return adj

    def Gradient(self, cache_blocking=None,
                 auto_tuning=False, dle='advanced', dse='advanced',
                 compiler=None, free_surface=False):
        """FWI gradient from back-propagation of a shot record
        and input forward wavefield
        """
        nt, nrec = self.data.shape
        ndim = len(self.model.shape)
        h = self.model.get_spacing()
        dtype = self.model.dtype
        nbpml = self.model.nbpml

        # Create receiver symbol
        rec = Receiver(name='rec', ntime=self.data.nt,
                       coordinates=self.data.receiver_coords)
        # Gradient symbol
        grad = DenseData(name="grad", shape=self.model.shape_domain,
                         dtype=self.model.dtype)

        # Create the forward wavefield
        v = TimeData(name="v", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)
        u = TimeData(name="u", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype, save=True)
        # Execute operator and return wavefield and receiver data
        gradop = GradientOperator(self.model, v, grad, rec, u, self.data,
                                  time_order=self.t_order, spc_order=self.s_order,
                                  cache_blocking=cache_blocking, dse=dse,
                                  dle=dle, tsave=self.tsave,
                                  free_surface=free_surface)
        return gradop

    def Born(self, save=False, cache_blocking=None,
             auto_tuning=False, dse='advanced',
             dle='advanced', compiler=None, free_surface=False):
        """Linearized modelling of one or multiple point source from
        an input model perturbation.
        """
        nt, nrec = self.data.shape
        nsrc = self.source.shape[1]
        ndim = len(self.model.shape)
        h = self.model.get_spacing()
        dtype = self.model.dtype
        nbpml = self.model.nbpml

        # Create source symbol
        src = PointSource(name='src', ntime=self.source.nt,
                          coordinates=self.source.receiver_coords)
        # Create receiver symbol
        Linrec = Receiver(name='Lrec', ntime=self.data.nt,
                          coordinates=self.data.receiver_coords)

        # Create the forward wavefield
        u = TimeData(name="u", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)
        du = TimeData(name="du", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)

        dm = DenseData(name="dm", shape=self.model.shape_domain, dtype=self.model.dtype)
        # Execute operator and return wavefield and receiver data
        born = BornOperator(self.model, u, du, src, Linrec, dm, self.data,
                            time_order=self.t_order, spc_order=self.s_order,
                            cache_blocking=cache_blocking, dse=dse,
                            dle=dle, free_surface=free_surface)
        return born


