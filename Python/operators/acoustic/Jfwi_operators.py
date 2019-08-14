from sympy import Eq, expand, Function, solve, symbols

from devito import t, time, x, y, z, Dimension
from devito.interfaces import DenseData, TimeData, Forward, Backward
from devito.foreign import Operator
from numpy.random import randint


def acoustic_laplacian(v, rho):
    # Derive stencil from symbolic equation
    if rho is not None:
        if isinstance(rho, DenseData):
            if len(v.shape[:-1]) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            if len(v.shape[:-1]) == 3:
                Lap = (1/rho * v.dx2 +
                       1/rho * v.dy2 +
                       1/rho * v.dz2)
            else:
                Lap = (1/rho * v.dx2 +
                       1/rho * v.dy2)
    else:
        Lap = v.laplace
        rho = 1
    return Lap, rho


def ForwardOperator(model, u, src, rec, data, q, time_order=2, spc_order=6,
                    save=False, tsave=4.0, free_surface=False, **kwargs):
    nt = data.shape[0]
    dt = model.critical_dt
    s = t.spacing
    m, damp, rho = model.m, model.damp, model.rho
    Lap, rho = acoustic_laplacian(u, rho)
    # Derive stencil from symbolic equation
    eqn = m / rho * u.dt2 - Lap + damp * u.dt + q
    # stencil = solve(eqn, u.forward)[0]
    stencil = solve(eqn, u.forward, rational=False)[0]
    # Add substitutions for spacing (temporal and spatial)
    subs = dict([(s, dt)] + [(i.spacing, model.get_spacing()[j]) for i, j
                             in zip(u.indices[1:], range(len(model.shape)))])
    stencils = [Eq(u.forward,stencil)]
    # Create stencil expressions for operator, source and receivers
    ti = u.indices[0]
    src_term = src.inject(field=u.forward, offset=model.nbpml,
                          expr=rho * src * dt**2 / m)
    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u, offset=model.nbpml)
    stencils = stencils + src_term + rec_term
    if save:
        nsave = int(nt/(tsave/dt) +1)
        rate = int(nt/nsave)+1
        usave = TimeData(name="usave", shape=model.shape_domain, time_dim=nt,
                         time_order=2, space_order=spc_order, save=True,
                         dtype=model.dtype)
        stencils += [Eq(usave.subs(usave.indices[0], Function('INT')(time/rate)), u)]
    
    if free_surface:
        fs = Dimension(name="fs", size = model.nbpml)
        stencils+= [Eq(u.forward.subs({u.indices[-1]: fs}), -u.forward.subs({u.indices[-1] :2*model.nbpml - fs}))]
    
    dse = kwargs.get('dse', 'advanced')
    dle = kwargs.get('dle', 'advanced')
    op = Operator(stencils, subs=subs, dse=dse, dle=dle,
                       time_axis=Forward, name="Forward%s" % randint(1e5),
                       profiler=False, external=True)
    
    return op


def AdjointOperator(model, v, srca, rec, data, time_order=2, spc_order=6,
                    save=False, free_surface=False, **kwargs):
    nt = data.shape[0]
    dt = model.critical_dt
    s = t.spacing
    m, damp, rho = model.m, model.damp, model.rho
    # Derive stencil from symbolic equation
    Lap, rho = acoustic_laplacian(v, rho)
    
    # Create the stencil by hand instead of calling numpy solve for speed purposes
    # Simple linear solve of a u(t+dt) + b u(t) + c u(t-dt) = L for u(t+dt)
    eqn = m / rho * v.dt2 - Lap - damp * v.dt
    stencil = solve(eqn, v.backward, rational=False)[0]
    # Add substitutions for spacing (temporal and spatial)
    subs = dict([(s, dt)] + [(i.spacing, model.get_spacing()[j]) for i, j
                             in zip(v.indices[1:], range(len(model.shape)))])
    dse = kwargs.get('dse', 'advanced')
    dle = kwargs.get('dle', 'advanced')
    
    # Create stencil expressions for operator, source and receivers
    eqn = Eq(v.backward, stencil)
    
    # Construct expression to inject receiver values
    ti = v.indices[0]
    receivers = rec.inject(field=v.backward, offset=model.nbpml,
                           expr=rho * rec * dt**2 / m)
    
    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v, offset=model.nbpml)
    stencils = [eqn] + source_a + receivers
    
    if free_surface:
        fs = Dimension(name="fs", size = model.nbpml)
        stencils+= [Eq(v.backward.subs({v.indices[-1]: fs}), -v.backward.subs({v.indices[-1]:2*model.nbpml - fs}))]
    
    op = Operator(stencils, subs=subs, dse=dse, dle=dle,
                       time_axis=Backward, name="Adjoint%s" % randint(1e5),
                       profiler=False, external=True)
    return op


def GradientOperator(model, v, grad, rec, u, data,
                     time_order=2, spc_order=6, tsave=4.0,
                     free_surface=False, **kwargs):
    """
    Class to setup the gradient operator in an acoustic media
    
    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    nt = data.shape[0]
    s = t.spacing
    dt = model.critical_dt
    m, damp, rho = model.m, model.damp, model.rho
    
    Lap, rho = acoustic_laplacian(v, rho)
    # Derive stencil from symbolic equation
    eqn = m / rho * v.dt2 - Lap - damp * v.dt
    stencil = solve(eqn, v.backward, rational=False)[0]
    nsave = int(nt/(tsave/dt) +1)
    rate = int(nt/nsave)+1
    gradient_update = Eq(grad, grad - ((time%(Function('INT')(rate)))<1) * u.subs(u.indices[0], Function('INT')(time/rate))* v.dt2 / rho)
    # Add substitutions for spacing (temporal and spatial)
    subs = dict([(s, dt)] + [(i.spacing, model.get_spacing()[j]) for i, j
                             in zip(v.indices[1:], range(len(model.shape)))])
    dse = kwargs.get('dse', 'advanced')
    dle = kwargs.get('dle', 'advanced')
    
    # Create stencil expressions for operator, source and receivers
    eqn = Eq(v.backward, stencil)
    # Add expression for receiver injection
    ti = v.indices[0]
    receivers = rec.inject(field=v.backward, offset=model.nbpml,
                           expr=rho * rec * dt * dt / m)
    stencils = [eqn] + receivers + [gradient_update]
    if free_surface:
        fs = Dimension(name="fs", size = model.nbpml)
        stencils+= [Eq(v.backward.subs({v.indices[-1]: fs}), -v.backward.subs({v.indices[-1]:2*model.nbpml - fs}))]
    op = Operator(stencils, subs=subs, dse=dse, dle=dle,
                       time_axis=Backward, name="Gradient%s" % randint(1e5),
                       profiler=False, external=True)
    
    return op

def BornOperator(model, u, du, src, Linrec, dm, data,
                 time_order=2, spc_order=6, save=False,
                 free_surface=False, **kwargs):
    """
    Class to setup the linearized modelling operator in an acoustic media
    
    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: dmin : square slowness perturbation
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    nt = data.shape[0]
    s = t.spacing
    dt = model.critical_dt
    m, damp, rho = model.m, model.damp, model.rho
    Lap, rho = acoustic_laplacian(u, rho)
    LapU, _ = acoustic_laplacian(du, rho)
    # Derive stencils from symbolic equation
    first_eqn = m / rho * u.dt2 - Lap + damp * u.dt
    first_stencil = solve(first_eqn, u.forward, rational=False)[0]
    second_eqn = m / rho * du.dt2 - LapU + damp * du.dt + dm / rho * u.dt2
    second_stencil = solve(second_eqn, du.forward, rational=False)[0]
    
    # Add substitutions for spacing (temporal and spatial)
    subs = dict([(s, dt)] + [(i.spacing, model.get_spacing()[j]) for i, j
                             in zip(u.indices[1:], range(len(model.shape)))])
    
    # Add Born-specific updates and resets
    dse = kwargs.get('dse', 'advanced')
    dle = kwargs.get('dle', 'advanced')
    
    # Create stencil expressions for operator, source and receivers
    eqn1 = [Eq(u.forward, first_stencil)]
    eqn2 = [Eq(du.forward, second_stencil)]
    # Add source term expression for u
    ti = u.indices[0]
    source = src.inject(field=u.forward, offset=model.nbpml,
                        expr=rho * src * dt * dt / m)
    
    # Create receiver interpolation expression from U
    receivers = Linrec.interpolate(expr=du, offset=model.nbpml)
    stencils = eqn1 + source + eqn2 + receivers
    if free_surface:
        fs = Dimension(name="fs", size = model.nbpml)
        stencils+= [Eq(u.forward.subs({u.indices[-1]: fs}), -u.forward.subs({u.indices[-1]:2*model.nbpml - fs}))]
        stencils+= [Eq(du.forward.subs({du.indices[-1]: fs}), -du.forward.subs({du.indices[-1]:2*model.nbpml - fs}))]
    op = Operator(stencils, subs=subs, dse=dse, dle=dle,
                  time_axis=Forward, name="Born%s" % randint(1e5),
                  profiler=False, external=True)
    return op

