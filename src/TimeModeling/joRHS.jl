############################################################
# joRHS ## #################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export joRHS, joRHSexception

############################################################

struct joRHS{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
	name::String
	m::Integer
	n::Integer
	info::Info
	geometry::Geometry
	data
end

mutable struct joRHSexception <: Exception
	msg :: String
end


############################################################

## outer constructors

"""
    joRHS
        name::String
        m::Integer
        n::Integer
        info::Info
        geometry::Geometry
        data

Abstract sparse vector for right-hand-sides of the modeling operators. The `joRHS` vector has the\\
dimensions of the full time history of the wavefields, but contains only the data defined at the \\
source or receiver positions (i.e. wavelets or shot records).

Constructor
==========

    joRHS(info, geometry, data)

Examples
========

Assuming `Pr` and `Ps` are projection operators of type `joProjection` and `dobs` and `q` are\\
seismic vectors of type `joData`, then a `joRHS` vector can be created as follows:

    rhs = Pr'*dobs    # right-hand-side with injected observed data
    rhs = Ps'*q    # right-hand-side with injected wavelet

"""
function joRHS(info,geometry,data;vDT::DataType=Float32)
	vDT == Float32 || throw(joRHSexception("Domain type not supported"))
	# length of vector
	m = info.n * sum(info.nt)
	n = 1
	return joRHS{Float32}("joRHS",m,n,info,geometry,data)
end

####################################################################
## overloaded Base functions

# conj(jo)
conj{vDT}(A::joRHS{vDT}) =
	joRHS{vDT}("conj("*A.name*")",A.m,A.n,A.info,A.geometry,A.data)

# transpose(jo)
transpose{vDT}(A::joRHS{vDT}) =
	joRHS{vDT}(""*A.name*".'",A.n,A.m,A.info,A.geometry,A.data)
   
# ctranspose(jo)
ctranspose{vDT}(A::joRHS{vDT}) =
	joRHS{vDT}(""*A.name*"'",A.n,A.m,A.info,A.geometry,A.data)

####################################################################

# +(joRHS,joRHS)
function +{avDT,bvDT}(A::joRHS{avDT}, B::joRHS{bvDT})

	# Error checking
	size(A) == size(B) || throw(joRHSexception("shape mismatch"))
	compareInfo(A.info, B.info) == true || throw(joRHSexception("info mismatch"))
	isequal(A.geometry.nt,B.geometry.nt) == true || throw(joRHSexception("sample number mismatch"))
	isequal(A.geometry.dt,B.geometry.dt) == true || throw(joRHSexception("sample interval mismatch"))
	isequal(A.geometry.t,B.geometry.t) == true || throw(joRHSexception("recording time mismatch"))

	# Size
	m = A.info.n * sum(A.info.nt)
	n = 1

	# merge geometries and data
	xloc = Array{Any}(A.info.nsrc)
	yloc = Array{Any}(A.info.nsrc)
	zloc = Array{Any}(A.info.nsrc)
	dt = Array{Any}(A.info.nsrc)
	nt = Array{Any}(A.info.nsrc)
	t = Array{Any}(A.info.nsrc)
	data = Array{Any}(A.info.nsrc)

	for j=1:A.info.nsrc
		xloc[j] = [vec(A.geometry.xloc[j])' vec(B.geometry.xloc[j])']
		yloc[j] = [vec(A.geometry.yloc[j])' vec(B.geometry.yloc[j])']
		zloc[j] = [vec(A.geometry.zloc[j])' vec(B.geometry.zloc[j])']
		dt[j] = A.geometry.dt[j]
		nt[j] = A.geometry.nt[j]
		t[j] = A.geometry.t[j]
		data[j] = [A.data[j] B.data[j]]
	end
	geometry = Geometry(xloc,yloc,zloc,dt,nt,t)
	nvDT = promote_type(avDT,bvDT)

	return joRHS{nvDT}("joRHS",m,n,A.info,geometry,data)
end

# -(joRHS,joRHS)
function -{avDT,bvDT}(A::joRHS{avDT}, B::joRHS{bvDT})

	# Error checking
	size(A) == size(B) || throw(joRHSexception("shape mismatch"))
	compareInfo(A.info, B.info) == true || throw(joRHSexception("info mismatch"))
	isequal(A.geometry.nt,B.geometry.nt) == true || throw(joRHSexception("sample number mismatch"))
	isequal(A.geometry.dt,B.geometry.dt) == true || throw(joRHSexception("sample interval mismatch"))
	isequal(A.geometry.t,B.geometry.t) == true || throw(joRHSexception("recording time mismatch"))

	# Size
	m = A.info.n * sum(A.info.nt)
	n = 1

	# merge geometries and data
	xloc = Array{Any}(A.info.nsrc)
	yloc = Array{Any}(A.info.nsrc)
	zloc = Array{Any}(A.info.nsrc)
	dt = Array{Any}(A.info.nsrc)
	nt = Array{Any}(A.info.nsrc)
	t = Array{Any}(A.info.nsrc)
	data = Array{Any}(A.info.nsrc)

	for j=1:A.info.nsrc
		xloc[j] = [vec(A.geometry.xloc[j])' vec(B.geometry.xloc[j])']
		yloc[j] = [vec(A.geometry.yloc[j])' vec(B.geometry.yloc[j])']
		zloc[j] = [vec(A.geometry.zloc[j])' vec(B.geometry.zloc[j])']
		dt[j] = A.geometry.dt[j]
		nt[j] = A.geometry.nt[j]
		t[j] = A.geometry.t[j]
		data[j] = [A.data[j] -B.data[j]]
	end
	geometry = Geometry(xloc,yloc,zloc,dt,nt,t)
	nvDT = promote_type(avDT,bvDT)

	return joRHS{nvDT}("joRHS",m,n,A.info,geometry,data)
end




