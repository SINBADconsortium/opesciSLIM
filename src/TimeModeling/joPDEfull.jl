############################################################
# joPDEfull ################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export joPDEfull, joPDEfullException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps, 
# i.e. it includes source and receiver projections

struct joPDEfull{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
	info::Info
	model::ModelAll
	srcGeometry::Geometry
	recGeometry::Geometry
	options::Options
	fop::Function              # forward
	fop_T::Nullable{Function}  # transpose
end

mutable struct joPDEfullException <: Exception
	msg :: String
end


############################################################
## Constructor

function joModeling(info::Info,model::ModelAll, srcGeometry::Geometry, recGeometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
	(DDT == Float32 && RDT == Float32) || throw(joPDEfullException("Domain and range types not supported"))

	# Determine dimensions
	if typeof(recGeometry) == GeometryOOC
		m = sum(recGeometry.nsamples)
	else
		m = 0
		for j=1:info.nsrc m += length(recGeometry.xloc[j])*recGeometry.nt[j] end

	end
	if typeof(srcGeometry) == GeometryOOC
		n = sum(srcGeometry.nsamples)
	else
		n = 0
		for j=1:info.nsrc n += length(srcGeometry.xloc[j])*srcGeometry.nt[j] end
	end

	if info.nsrc > 1
		srcnum = 1:info.nsrc
	else
		srcnum = 1
	end

	return F = joPDEfull{Float32,Float32}("Proj*F*Proj'", m, n, info, model, srcGeometry, recGeometry, options,
							  src -> time_modeling(model, srcGeometry, src.data, recGeometry, [], [], srcnum, 'F', 1, options),
							  rec -> time_modeling(model, srcGeometry, [], recGeometry, rec.data, [], srcnum, 'F', -1, options),
							  )
end


############################################################
## overloaded Base functions

# conj(joPDEfull)
conj{DDT,RDT}(A::joPDEfull{DDT,RDT}) =
	joPDEfull{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
		get(A.fop),
		A.fop_T
		)

# transpose(joPDEfull)
transpose{DDT,RDT}(A::joPDEfull{DDT,RDT}) =
	joPDEfull{DDT,RDT}("Proj*F'*Proj'",A.n,A.m,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
		get(A.fop_T),
		A.fop
		)

# ctranspose(joPDEfull)
ctranspose{DDT,RDT}(A::joPDEfull{DDT,RDT}) =
	joPDEfull{DDT,RDT}("Proj*F'*Proj'",A.n,A.m,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
		get(A.fop_T),
		A.fop
		)

############################################################
## overloaded Base *(...joPDEfull...)

# *(joPDEfull,joData)
function *{ADDT,ARDT,vDT}(A::joPDEfull{ADDT,ARDT},v::joData{vDT})
	A.n == size(v,1) || throw(joPDEfullException("shape mismatch"))
	if compareGeometry(A.srcGeometry,v.geometry) == false && compareGeometry(A.recGeometry,v.geometry) == false
		throw(joPDEfullException("geometry mismatch"))
	end
	jo_check_type_match(ADDT,vDT,join(["DDT for *(joPDEfull,joData):",A.name,typeof(A),vDT]," / "))
	V = A.fop(v)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joPDEfull,joData):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(num,joPDEfull)
function *{ADDT,ARDT}(a::Number,A::joPDEfull{ADDT,ARDT})
	return joPDEfull{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
		v1 -> jo_convert(ARDT,a*A.fop(v1),false),
		v2 -> jo_convert(ADDT,a*A.fop_T(v2),false)
		)
end


############################################################
## overloaded Bases +(...joPDEfull...), -(...joPDEfull...)

# +(joPDEfull,num)
function +{ADDT,ARDT}(A::joPDEfull{ADDT,ARDT},b::Number)
	return joPDE{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
		v1 -> A.fop(v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
		v2 -> get(A.fop_T)(v2)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v2
		)
end

# -(joPDEfull,num)
function -{ADDT,ARDT}(A::joPDEfull{ADDT,ARDT},b::Number)
	return joPDE{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
		v1 -> A.fop(v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
		v2 -> get(A.fop_T)(v2)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v2
		)
end

# -(joPDEfull)
-{DDT,RDT}(A::joPDEfull{DDT,RDT}) =
	joPDEfull{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
		v1 -> -A.fop(v1),
		v2 -> -get(A.fop_T)(v2)
		)


############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample{ADDT,ARDT}(F::joPDEfull{ADDT,ARDT}, srcnum)

	srcGeometry = subsample(F.srcGeometry,srcnum)		# Geometry of subsampled data container
	recGeometry = subsample(F.recGeometry,srcnum)

	info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
	return joModeling(info, F.model, srcGeometry, recGeometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

getindex(F::joPDEfull,a) = subsample(F,a)



