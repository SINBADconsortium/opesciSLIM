############################################################
# joJacobian ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export joJacobian, joJacobianException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps, 
# i.e. it includes source and receiver projections
struct joJacobian{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
	name::String
	m::Integer
	n::Integer
	info::Info
	model::ModelAll
	srcGeometry::Geometry
	recGeometry::Geometry
	source
	options::Options
	fop::Function              # forward
	fop_T::Nullable{Function}  # transpose
end


mutable struct joJacobianException <: Exception
	msg :: String
end

############################################################
## Constructor
"""
    joJacobian(F,q)

Create a linearized modeling operator from the non-linear modeling operator `F` and \\
the source `q`. `F` is a full modeling operator including source/receiver projections.

Examples
========

1) `F` is a modeling operator without source/receiver projections:

    J = joJacobian(Pr*F*Ps',q)

2) `F` is the combined operator `Pr*F*Ps'`:

    J = joJacobian(F,q)

"""
function joJacobian(F::joPDEfull, source::joData; DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
	compareGeometry(F.srcGeometry, source.geometry) == true || joJacobianException("Source geometry mismatch")
	(DDT == Float32 && RDT == Float32) || throw(joJacobianException("Domain and range types not supported"))
	if typeof(F.recGeometry) == GeometryOOC
		m = sum(F.recGeometry.nsamples)
	else
		m = 0
		for j=1:F.info.nsrc m += length(F.recGeometry.xloc[j])*F.recGeometry.nt[j] end
	end
	n = F.info.n
	if F.info.nsrc > 1
		srcnum = 1:F.info.nsrc
	else
		srcnum = 1
	end
	return J = joJacobian{Float32,Float32}("linearized wave equation", m, n, F.info, F.model, F.srcGeometry, F.recGeometry, source.data, F.options,
                                           v -> time_modeling(F.model, F.srcGeometry, source.data, F.recGeometry, [], v, srcnum, 'J', 1, F.options),
                                           w -> time_modeling(F.model, F.srcGeometry, source.data, F.recGeometry, w.data, [], srcnum, 'J', -1, F.options)
										   )
end


############################################################
## overloaded Base functions

# conj(joJacobian)
conj{DDT,RDT}(A::joJacobian{DDT,RDT}) =
	joJacobian{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
		get(A.fop),
		A.fop_T
		)

# transpose(joJacobian)
transpose{DDT,RDT}(A::joJacobian{DDT,RDT}) =
	joJacobian{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
		get(A.fop_T),
		A.fop
		)

# ctranspose(joJacobian)
ctranspose{DDT,RDT}(A::joJacobian{DDT,RDT}) =
	joJacobian{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
		get(A.fop_T),
		A.fop
		)

############################################################
## overloaded Base *(...joJacobian...)

# *(joJacobian,vec)
function *{ADDT,ARDT,vDT}(A::joJacobian{ADDT,ARDT},v::AbstractVector{vDT})
	A.n == size(v,1) || throw(joJacobianException("shape mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(joJacobian,vec):",A.name,typeof(A),vDT]," / "))
	V = A.fop(v)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joJacobian,vec):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(joJacobian,joData)
function *{ADDT,ARDT,vDT}(A::joJacobian{ADDT,ARDT},v::joData{vDT})
	A.n == size(v,1) || throw(joJacobianException("shape mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(joJacobian,joData):",A.name,typeof(A),vDT]," / "))
	compareGeometry(A.recGeometry,v.geometry) == true || throw(joJacobianException("Geometry mismatch"))
	V = A.fop(v)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joJacobian,joData):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(num,joJacobian)
function *{ADDT,ARDT}(a::Number,A::joJacobian{ADDT,ARDT})
	return joJacobian{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
								v1 -> jo_convert(ARDT,a*A.fop(v1),false),
								v2 -> jo_convert(ADDT,a*A.fop_T(v2),false)
								)
end

############################################################
## overloaded Bases +(...joJacobian...), -(...joJacobian...)

# +(joJacobian,num)
function +{ADDT,ARDT}(A::joJacobian{ADDT,ARDT},b::Number)
	return joJacobian{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
							v1 -> A.fop(v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
							v2 -> get(A.fop_T)(v2)+joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
							)
end

# -(joJacobian,num)
function -{ADDT,ARDT}(A::joJacobian{ADDT,ARDT},b::Number)
	return joJacobian{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
							v1 -> A.fop(v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
							v2 -> get(A.fop_T)(v2)-joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
							)
end

# -(joJacobian)
-{DDT,RDT}(A::joJacobian{DDT,RDT}) =
	joJacobian{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
					v1 -> -A.fop(v1),
					v2 -> -get(A.fop_T)(v2)
					)

############################################################
## Additional overloaded functions

# Subsample Jacobian
function subsample{ADDT,ARDT}(J::joJacobian{ADDT,ARDT}, srcnum)
	
	srcGeometry = subsample(J.srcGeometry,srcnum)		# Geometry of subsampled data container
	recGeometry = subsample(J.recGeometry,srcnum)

	info = Info(J.info.n, length(srcnum), J.info.nt[srcnum])
	Fsub = joModeling(info, J.model, srcGeometry, recGeometry; options=J.options)
	qsub = joData(srcGeometry, J.source[srcnum])
	return joJacobian(Fsub, qsub; DDT=ADDT, RDT=ARDT)
end

getindex(J::joJacobian,a) = subsample(J,a)



