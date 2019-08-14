# Module with functions for time-domain modeling and inversion using OPESCI/devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
#

module TimeModeling
using PyCall, JOLI, SeisIO
@pyimport numpy as np
@pyimport Jcontainers as ct
@pyimport JAcoustic_codegen as ac
@pyimport JTTI_codegen as tc
@pyimport devito as dvt
import Base.*, Base./, Base.+, Base.-, Base.ctranspose, Base.conj, Base.vcat, Base.vec, Base.dot, Base.norm, Base.abs, Base.getindex


#############################################################################
# Containers
include("ModelStructure.jl")	# model container
include("InfoStructure.jl")	# basic information required by all operators
include("GeometryStructure.jl")	# source or receiver setup, recording time and sampling
include("OptionsStructure.jl")
include("auxiliaryFunctions.jl")

#############################################################################
# Abstract vectors
include("joRHS.jl")	# RHS to be multiplied with linear operator
include("joData.jl")	# Julia data container

#############################################################################
# PDE solvers
include("time_modeling_serial.jl")	# forward/adjoint linear/nonlinear modeling
include("time_modeling_serial_rho.jl")	# forward/adjoint linear/nonlinear modeling w/ density
include("time_modeling_parallel.jl")	# parallelization for modeling
include("fwi_objective_serial.jl")	# FWI objective function value and gradient
include("fwi_objective_serial_rho.jl")  # with density
include("fwi_objective_parallel.jl")	# parallelization for FWI gradient

#############################################################################
# Linear operators
include("joModeling.jl")	# nonlinear modeling operator F (no projection operators)
include("joProjection.jl")	# source/receiver projection operator
include("joPDEfull.jl")	# modeling operator with source and receiver projection: P*F*P'
include("joPDE.jl")	# modeling operator with lhs projection only: P*F
include("joJacobian.jl")	# linearized modeling operator J

end
