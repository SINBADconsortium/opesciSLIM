# Model structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

const IntTuple = Union{Tuple{Integer,Integer}, Tuple{Integer,Integer,Integer}}
const RealTuple = Union{Tuple{Real,Real}, Tuple{Real,Real,Real}}

export Model, ModelRho

# Object for velocity/slowness models
mutable struct Model
	n::IntTuple
	d::RealTuple
	o::RealTuple
	nb::Integer	# number of absorbing boundaries points on each side
	m	# slowness squared
end

mutable struct ModelRho
	n::IntTuple
	d::RealTuple
	o::RealTuple
	nb::Integer
	m	# slowness squared
	rho
end

"""
    Model
        n::IntTuple
        d::RealTuple
        o::RealTuple
        nb::Integer
        m::Array
        rho::Array


Model structure for seismic velocity models.

`n`: number of gridpoints in (x,y,z) for 3D or (x,z) for 2D

`d`: grid spacing in (x,y,z) or (x,z) (in meters)

`o`: origin of coordinate system in (x,y,z) or (x,z) (in meters)

`nb`: number of absorbing boundary points in each direction

`m`: velocity model in slowness squared (s^2/km^2)

`rho`: density in (g/cm^3), (optional, default is rho=1)


Constructor
===========

The parameters `n`, `d`, `o` and `m` are mandatory. All other input arguments are optional:

    Model(n, d, o, m; nb=40, rho=ones(n))

For zero Thomsen parameters, the acoustic wave equation is solved. Otherwise the TTI equation is solved.

"""
Model(n::IntTuple,d::RealTuple,o::RealTuple,m;nb=40) = Model(n,d,o,nb,m)
Model(n::IntTuple,d::RealTuple,o::RealTuple,m,rho;nb=40) = ModelRho(n,d,o,nb,m,rho)

function Model(path::String; parameter="m")
	A = load(path)
	if parameter=="m"
		return Model(A["n"], A["d"], A["o"], A["m"])
	elseif parameter=="m0"
		return Model(A["n"], A["d"], A["o"], A["m0"])
	elseif parameter=="dm"
		return Model(A["n"], A["d"], A["o"], A["dm"])
	end
end

const ModelAll = Union{Model,ModelRho}
