############################################################
# joPDE ####################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export joPDE, joPDEexception, joPDEadjoint, joPDEadjointException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps, 
# i.e. it includes source and receiver projections

# Forward
struct joPDE{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
	info::Info
	model::ModelAll
	geometry::Geometry
	options::Options
	fop::Function              # forward
	fop_T::Nullable{Function}  # transpose
end

mutable struct joPDEexception <: Exception
	msg :: String
end

# Adjoint
struct joPDEadjoint{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
	name::String
	m::Integer
	n::Integer
	info::Info
	model::ModelAll
	geometry::Geometry
	options::Options
	fop::Function              # forward
	fop_T::Nullable{Function}  # transpose
end

mutable struct joPDEadjointException <: Exception
	msg :: String
end



############################################################
## outer constructors

function joPDE(name::String,info::Info,model::ModelAll, geometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
	(DDT == Float32 && RDT == Float32) || throw(joPDEexception("Domain and range types not supported"))
	if typeof(geometry) == GeometryOOC
		m = sum(geometry.nsamples)
	else
		m = 0
		for j=1:info.nsrc m += length(geometry.xloc[j])*geometry.nt[j] end
	end
	n = info.n * sum(info.nt)
	if info.nsrc > 1
		srcnum = 1:info.nsrc
	else
		srcnum = 1
	end
	return F = joPDE{Float32,Float32}(name, m, n, info, model, geometry, options,
							  src -> time_modeling(model, src.geometry, src.data, geometry, [], [], srcnum, 'F', 1, options),
							  rec -> time_modeling(model, geometry, [], rec.geometry, rec.data, [], srcnum, 'F', -1, options)
							  )
end

function joPDEadjoint(name::String,info::Info,model::ModelAll, geometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
	(DDT == Float32 && RDT == Float32) || throw(joPDEadjointException("Domain and range types not supported"))
	if typeof(geometry) == GeometryOOC
		m = sum(geometry.nsamples)
	else
		m = 0
		for j=1:info.nsrc m += length(geometry.xloc[j])*geometry.nt[j] end
	end
	n = info.n * sum(info.nt)
	if info.nsrc > 1
		srcnum = 1:info.nsrc
	else
		srcnum = 1
	end
	return F = joPDEadjoint{Float32,Float32}(name, m, n, info, model, geometry, options,
									 rec -> time_modeling(model, geometry, [], rec.geometry, rec.data, [], srcnum, 'F', -1, options),
									 src -> time_modeling(model, src.geometry, src.data, geometry, [], [], srcnum, 'F', 1, options),
								     )
end


############################################################
## overloaded Base functions

# conj(joPDE)
conj{DDT,RDT}(A::joPDE{DDT,RDT}) =
	joPDE{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
	get(A.fop),
		A.fop_T
		)

# transpose(jo)
transpose{DDT,RDT}(A::joPDE{DDT,RDT}) =
	joPDEadjoint{DDT,RDT}(A.name,A.n,A.m,A.info,A.model,A.geometry,A.options,
		get(A.fop_T),
		A.fop
		)

# ctranspose(jo)
ctranspose{DDT,RDT}(A::joPDE{DDT,RDT}) =
	joPDEadjoint{DDT,RDT}(A.name,A.n,A.m,A.info,A.model,A.geometry,A.options,
		get(A.fop_T),
		A.fop
		)

# conj(jo)
conj{DDT,RDT}(A::joPDEadjoint{DDT,RDT}) =
    joPDEadjoint{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
        get(A.fop),
        A.fop_T
        )

# transpose(jo)
transpose{DDT,RDT}(A::joPDEadjoint{DDT,RDT}) =
	joPDE{DDT,RDT}(A.name,A.n,A.m,A.info,A.model,A.geometry,A.options,
		get(A.fop_T),
		A.fop
		)

# ctranspose(jo)
ctranspose{DDT,RDT}(A::joPDEadjoint{DDT,RDT}) =
	joPDE{DDT,RDT}(A.name,A.n,A.m,A.info,A.model,A.geometry,A.options,
		get(A.fop_T),
		A.fop
		)

############################################################
## overloaded Base *(...jo...)

# *(joPDE,joRHS)
function *{ADDT,ARDT,vDT}(A::joPDE{ADDT,ARDT},v::joRHS{vDT})
	A.n == size(v,1) || throw(joPDEexception("shape mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(joPDE,joRHS):",A.name,typeof(A),vDT]," / "))
	V = A.fop(v)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joPDE,joData):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

function *{ADDT,ARDT,vDT}(A::joPDEadjoint{ADDT,ARDT},v::joRHS{vDT})
	A.n == size(v,1) || throw(joPDEadjointException("shape mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(joPDE,joRHS):",A.name,typeof(A),vDT]," / "))
	V = A.fop(v)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joPDE,joData):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(joPDE,joProjection)
function *{ARDT,BDDT,CDT}(A::joPDE{CDT,ARDT},B::joProjection{BDDT,CDT})
	A.n == size(B,1) || throw(joPDEexception("shape mismatch"))
	return joModeling(A.info,A.model,B.geometry,A.geometry;options=A.options,DDT=CDT,RDT=ARDT)
end

function *{ARDT,BDDT,CDT}(A::joPDEadjoint{CDT,ARDT},B::joProjection{BDDT,CDT})
	A.n == size(B,1) || throw(joPDEadjointException("shape mismatch"))
	return joModeling(A.info,A.model,A.geometry,B.geometry,options=A.options,DDT=CDT,RDT=ARDT)'
end

# *(num,joPDE)
function *{ADDT,ARDT}(a::Number,A::joPDE{ADDT,ARDT})
	return joPDE{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
		v1 -> jo_convert(ARDT,a*A.fop(v1),false),
		v2 -> jo_convert(ADDT,a*A.fop_T(v2),false)
		)
end

function *{ADDT,ARDT}(a::Number,A::joPDEadjoint{ADDT,ARDT})
	return joPDEadjoint{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
		v1 -> jo_convert(ARDT,a*A.fop(v1),false),
		v2 -> jo_convert(ADDT,a*A.fop_T(v2),false)
		)
end


############################################################
## overloaded Basees +(...joPDE...), -(...joPDE...)

# +(joPDE,num)
function +{ADDT,ARDT}(A::joPDE{ADDT,ARDT},b::Number)
	return joPDE{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.geometry,A.options,
		v1 -> A.fop(v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
		v2 -> A.fop_T(v2)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v2
		)
end

function +{ADDT,ARDT}(A::joPDEadjoint{ADDT,ARDT},b::Number)
	return joPDEadjoint{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.geometry,A.options,
		v1->A.fop(v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
		v2->A.fop_T(v2)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v2
		)
end

# -(joPDE,num)
function -{ADDT,ARDT}(A::joPDE{ADDT,ARDT},b::Number)
	return joPDE{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.geometry,A.options,
		v1 -> A.fop(v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
		v2 -> A.fop_T(v2)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v2
		)
end

function -{ADDT,ARDT}(A::joPDEadjoint{ADDT,ARDT},b::Number)
	return joPDEadjoint{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.geometry,A.options,
		v1->A.fop(v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
		v2->A.fop_T(v2)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v2
		)
end

# -(joPDE)
-{DDT,RDT}(A::joPDE{DDT,RDT}) =
	joPDE{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
		v1->-A.fop(v1),
		v2->-get(A.fop_T)(v2)
		)

-{DDT,RDT}(A::joPDEadjoint{DDT,RDT}) =
	joPDEadjoint{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
		v1->-A.fop(v1),
		v2->-get(A.fop_T)(v2)
		)


############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample{ADDT,ARDT}(F::joPDE{ADDT,ARDT}, srcnum)
	geometry = subsample(F.geometry,srcnum)		# Geometry of subsampled data container
	info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
	return joPDE(F.name, info, F.model, geometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

function subsample{ADDT,ARDT}(F::joPDEadjoint{ADDT,ARDT}, srcnum)
	geometry = subsample(F.geometry,srcnum)		# Geometry of subsampled data container
	info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
	return joPDEadjoint(F.name, info, F.model, geometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

getindex(F::joPDE,a) = subsample(F,a)
getindex(F::joPDEadjoint,a) = subsample(F,a)




