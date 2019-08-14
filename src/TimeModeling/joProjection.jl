############################################################
# joProjection #############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export joProjection, joProjectionException

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps, 
# i.e. it includes source and receiver projections
struct joProjection{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
	name::String
	m::Integer
	n::Integer
	info::Info
	geometry::Geometry
end


mutable struct joProjectionException <: Exception
	msg :: String
end


############################################################
## Constructor
"""
    joProjection(info, geometry)

Projection operator for sources/receivers to restrict or inject data at specified locations.\\
`info` is an `Info` structure and `geometry` is a `Geometry` structure with either source or\\
receiver locations.

Examples
========

`F` is a modeling operator of type `joModeling` and `q` is a seismic source of type `joData`:

    Pr = joProjection(info, rec_geometry)
    Ps = joProjection(info, q.geometry)

    dobs = Pr*F*Ps'*q
    qad = Ps*F'*Pr'*dobs

"""
function joProjection(info::Info, geometry::GeometryIC; DDT::DataType=Float32, RDT::DataType=DDT)
	(DDT == Float32 && RDT == Float32) || throw(joProjectionException("Domain and range types not supported"))
	m = 0
	for j=1:length(geometry.xloc)
		m += length(geometry.xloc[j])*geometry.nt[j]
	end
	n = info.n * sum(info.nt)

	return joProjection{Float32,Float32}("restriction operator",m,n,info,geometry)
end

function joProjection(info::Info, geometry::GeometryOOC; DDT::DataType=Float32, RDT::DataType=DDT)
	(DDT == Float32 && RDT == Float32) || throw(joProjectionException("Domain and range types not supported"))
	m = sum(geometry.nsamples)
	n = info.n * sum(info.nt)

	return joProjection{Float32,Float32}("restriction operator",m,n,info,geometry)
end


############################################################
## overloaded Base functions

# conj(joProjection)
conj{DDT,RDT}(A::joProjection{DDT,RDT}) =
	joProjection{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.geometry)

# transpose(joProjection)
transpose{DDT,RDT}(A::joProjection{DDT,RDT}) =
	joProjection{DDT,RDT}("injection operator",A.n,A.m,A.info,A.geometry)

# ctranspose(joProjection)
ctranspose{DDT,RDT}(A::joProjection{DDT,RDT}) =
	joProjection{DDT,RDT}("injection operator",A.n,A.m,A.info,A.geometry)

############################################################
## overloaded Base *(...joProjection...)

# *(joProjection,joData)
function *{ADDT,ARDT,vDT}(A::joProjection{ADDT,ARDT},v::joData{vDT})
	A.n == size(v,1) || throw(joProjectionException("shape mismatch"))
	compareGeometry(A.geometry,v.geometry) == true || throw(joProjectionException("geometry mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(joProjection,joData):",A.name,typeof(A),vDT]," / "))
	V = joRHS(A.info,v.geometry,v.data)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joProjection,joData):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(joProjection,joModeling)
function *{ARDT,BDDT,CDT}(A::joProjection{CDT,ARDT},B::joModeling{BDDT,CDT})
	A.n == size(B,1) || throw(joProjectionException("shape mismatch"))
	compareInfo(A.info, B.info) == true || throw(joProjectionException("info mismatch"))
	if typeof(A.geometry) == GeometryOOC
		m = sum(A.geometry.nsamples)
	else
		m = 0; for j=1:B.info.nsrc m+= length(A.geometry.xloc[j])*A.geometry.nt[j] end
	end
	n = B.info.n * sum(B.info.nt)
	return joPDE("joProjection*joModeling",B.info,B.model,A.geometry;options=B.options,DDT=CDT,RDT=ARDT)
end

function *{ARDT,BDDT,CDT}(A::joProjection{CDT,ARDT},B::joModelingAdjoint{BDDT,CDT})
	A.n == size(B,1) || throw(joProjectionException("shape mismatch"))
	compareInfo(A.info, B.info) == true || throw(joProjectionException("info mismatch"))
	if typeof(A.geometry) == GeometryOOC
		m = sum(A.geometry.nsamples)
	else
		m = 0; for j=1:B.info.nsrc m+= length(A.geometry.xloc[j])*A.geometry.nt[j] end
	end
	n = B.info.n * sum(B.info.nt)
	return joPDEadjoint("joProjection*joModelingAdjoint",B.info,B.model,A.geometry;options=B.options,DDT=CDT,RDT=ARDT)
end

############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample{ADDT,ARDT}(P::joProjection{ADDT,ARDT}, srcnum)
	geometry = subsample(P.geometry,srcnum)		# Geometry of subsampled data container
	info = Info(P.info.n, length(srcnum), P.info.nt[srcnum])
	return joProjection(info, geometry;DDT=ADDT,RDT=ARDT)
end

getindex(P::joProjection,a) = subsample(P,a)

