############################################################
# joModeling ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export joModeling, joModelingException, joSetupModeling, joModelingAdjoint, joModelingAdjointException, joSetupModelingAdjoint, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps, 
# i.e. it includes source and receiver projections
struct joModeling{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
	name::String
	m::Integer
	n::Integer
	info::Info
	model::ModelAll
	options::Options
end

mutable struct joModelingException <: Exception
	msg :: String
end


struct joModelingAdjoint{DDT,RDT} <: joAbstractLinearOperator{DDT,RDT}
	name::String
	m::Integer
	n::Integer
	info::Info
	model::ModelAll
	options::Options
end

mutable struct joModelingAdjointException <: Exception
	msg :: String
end


############################################################
## outer constructors
"""
    joModeling(info, model; options=Options())
	joModeling(info, model, src_geometry, rec_geometry; options=Options())

Create seismic modeling operator for a velocity model given as a `Model` structure. `info` is an `Info` structure\\
containing necessary dimensions to set up the operator. The function also takes the source and receiver geometries\\
as additional input arguments, which creates a combined operator `joProjection*joModeling*joProjection'`.

Example
=======

`Pr` and `Ps` are projection operatos of type `joProjection` and\\
`q` is a data vector of type `joData`:

    F = joModeling(info, model)
    dobs = Pr*F*Ps'*q

	F = joModeling(info, model, q.geometry, rec\_geometry)
    dobs = F*q

"""
function joModeling(info::Info, model::ModelAll; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
	(DDT == Float32 && RDT == Float32) || throw(joModelingException("Domain and range types not supported"))
	m = info.n * sum(info.nt)
	n = m
	if info.nsrc > 1
		srcnum = 1:info.nsrc
	else
		srcnum = 1
	end
	return F = joModeling{Float32,Float32}("forward wave equation", m, n, info, model, options)
end

function joModelingAdjoint(info::Info, model::ModelAll; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
	(DDT == Float32 && RDT == Float32) || throw(joModelingAdjointException("Domain and range types not supported"))
	m = info.n * sum(info.nt)
	n = m
	if info.nsrc > 1
		srcnum = 1:info.nsrc
	else
		srcnum = 1
	end
	return F = joModelingAdjoint{Float32,Float32}("adjoint wave equation", m, n, info, model, options)
end


############################################################
## overloaded Base functions

# conj(jo)
conj{DDT,RDT}(A::joModeling{DDT,RDT}) =
	joModeling{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.options)

# transpose(jo)
transpose{DDT,RDT}(A::joModeling{DDT,RDT}) =
	joModelingAdjoint{DDT,RDT}("adjoint wave equation",A.n,A.m,A.info,A.model,A.options)

# ctranspose(jo)
ctranspose{DDT,RDT}(A::joModeling{DDT,RDT}) =
	joModelingAdjoint{DDT,RDT}("adjoint wave equation",A.n,A.m,A.info,A.model,A.options)

# conj(jo)
conj{DDT,RDT}(A::joModelingAdjoint{DDT,RDT}) =
	joModelingAdjoint{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.options)

# transpose(jo)
transpose{DDT,RDT}(A::joModelingAdjoint{DDT,RDT}) =
	joModeling{DDT,RDT}("forward wave equation",A.n,A.m,A.info,A.model,A.options)

# ctranspose(jo)
ctranspose{DDT,RDT}(A::joModelingAdjoint{DDT,RDT}) =
	joModeling{DDT,RDT}("forward wave equation",A.n,A.m,A.info,A.model,A.options)


############################################################
## Additional overloaded functions

# Subsample Modeling functino
function subsample(F::joModeling, srcnum)
	info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
	return joModeling(info, F.model, options=F.options)
end

getindex(F::joModeling,a) = subsample(F,a)





