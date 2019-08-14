############################################################
# joData # #################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export joData, joDataException, subsample, joData_to_SeisBlock

############################################################

# structure for seismic data as an abstract vector
mutable struct joData{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
	name::String
	m::Integer
	n::Integer
	nsrc::Integer
	geometry::Geometry
	data
end

mutable struct joDataException <: Exception
	msg :: String
end

############################################################

## outer constructors

"""
    joData
	    name::String
        m::Integer
        n::Integer
        nsrc::Integer
        geometry::Geometry
        data

Abstract vector for seismic data. This vector-like structure contains the geometry and data for either\\
receiver data (shot records) or source data (wavelets).

Constructors
============

Construct vector from `Geometry` structure and cell array of shot records or wavelets. The `data` keyword\\
can also be a single (non-cell) array, in which case the data is the same for all source positions:

    joData(geometry, data)

Construct vector for observed data from `SeisIO.SeisBlock`. `segy_depth_key` is the `SeisIO` keyword \\
that contains the receiver depth coordinate:

    joData(SeisIO.SeisBlock; segy_depth_key="RecGroupElevation")

Construct vector for observed data from out-of-core data container of type `SeisIO.SeisCon`:

    joData(SeisIO.SeisCon; segy_depth_key="RecGroupElevation")

Examples
========

(1) Construct data vector from `Geometry` structure and a cell array of shot records:
  
    dobs = joData(rec_geometry, shot_records)

(2) Construct data vector for a seismic wavelet (can be either cell arrays of individual\\
wavelets or a single wavelet as an array):

    q = joData(src_geometry, wavelet)

(3) Construct data vector from `SeisIO.SeisBlock` object:

    using SeisIO
    seis_block = segy_read("test_file.segy")
	dobs = joData(seis_block; segy_depth_key="RecGroupElevation")

(4) Construct out-of-core data vector from `SeisIO.SeisCon` object (for large SEG-Y files):

    using SeisIO
    seis_container = segy_scan("/path/to/data/directory","filenames",["GroupX","GroupY","RecGroupElevation","SourceDepth","dt"])
    dobs = joData(seis_container; segy_depth_key="RecGroupElevation")

"""
function joData(geometry::Geometry,data::Array; vDT::DataType=Float32)
	vDT == Float32 || throw(joDataException("Domain type not supported"))

	# length of vector
	n = 1
	if typeof(geometry) == GeometryOOC
		nsrc = length(geometry.container)
		m = sum(geometry.nsamples)
	else
		nsrc = length(geometry.xloc)
		m = 0
		for j=1:nsrc
			m += length(geometry.xloc[j])*geometry.nt[j]
		end
	end
	dataCell = Array{Array}(nsrc)
	for j=1:nsrc
		dataCell[j] = data
	end
	return joData{Float32}("Seismic data vector",m,n,nsrc,geometry,dataCell)
end

# constructor if data is passed as a cell array
function joData(geometry::Geometry,data::Union{Array{Any,1},Array{Array,1}}; vDT::DataType=Float32)
	vDT == Float32 || throw(joDataException("Domain and range types not supported"))

	# length of vector
	if typeof(geometry) == GeometryOOC
		nsrc = length(geometry.container)
		m = sum(geometry.nsamples)
	else
		nsrc = length(geometry.xloc)
		m = 0
		for j=1:nsrc
			m += length(geometry.xloc[j])*geometry.nt[j]
		end
	end
	n = 1
	return joData{Float32}("Seismic data vector",m,n,nsrc,geometry,data)
end


############################################################
# constructors from SEGY files or out-of-core containers

# contructor for in-core data container
function joData(data::SeisIO.SeisBlock; segy_depth_key="RecGroupElevation", vDT::DataType=Float32)
	vDT == Float32 || throw(joDataException("Domain and range types not supported"))

	# length of data vector
	src = get_header(data,"FieldRecord")
 	nsrc = length(unique(src))
	
	numTraces = get_header(data,"TraceNumber")[end] - get_header(data,"TraceNumber")[1] + 1
	numSamples = get_header(data,"ns")[end]
	m = numTraces*numSamples
	n = 1

	# extract geometry from data container
	geometry = Geometry(data; key="receiver", segy_depth_key=segy_depth_key)

	# fill data vector with pointers to data location
	dataCell = Array{Array}(nsrc)
	full_data = convert(Array{Float32,2},data.data)
	for j=1:nsrc
		traces = find(src .== unique(src)[j])
		dataCell[j] = full_data[:,traces]
	end

	return joData{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for in-core data container and given geometry
function joData(geometry::Geometry, data::SeisIO.SeisBlock; vDT::DataType=Float32)
	vDT == Float32 || throw(joDataException("Domain and range types not supported"))

	# length of data vector
	src = get_header(data,"FieldRecord")
 	nsrc = length(unique(src))
	
	numTraces = get_header(data,"TraceNumber")[end] - get_header(data,"TraceNumber")[1] + 1
	numSamples = get_header(data,"ns")[end]
	m = numTraces*numSamples
	n = 1

	# fill data vector with pointers to data location
	dataCell = Array{Array}(nsrc)
	for j=1:nsrc
		traces = find(src .== unique(src)[j])
		dataCell[j] = convert(Array{Float32,2},data.data[:,traces])
	end

	return joData{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from single container
function joData(data::SeisIO.SeisCon; segy_depth_key="RecGroupElevation", vDT::DataType=Float32)
	vDT == Float32 || throw(joDataException("Domain and range types not supported"))

	# length of data vector
	nsrc = length(data)
	numTraces = 0
	for j=1:nsrc
		numTraces += Int((data.blocks[j].endbyte - data.blocks[j].startbyte)/(240 + data.ns*4))
	end
	m = numTraces*data.ns
	n = 1

	# extract geometry from data container
	geometry = Geometry(data; key="receiver", segy_depth_key=segy_depth_key)

	# fill data vector with pointers to data location
	dataCell = Array{SeisIO.SeisCon}(nsrc)
	for j=1:nsrc
		dataCell[j] = split(data,j)
	end

	return joData{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for single out-of-core data container and given geometry
function joData(geometry::Geometry, data::SeisIO.SeisCon; vDT::DataType=Float32)
	vDT == Float32 || throw(joDataException("Domain and range types not supported"))

	# length of data vector
	nsrc = length(data)
	numTraces = 0
	for j=1:nsrc
		numTraces += Int((data.blocks[j].endbyte - data.blocks[j].startbyte)/(240 + data.ns*4))
	end
	m = numTraces*data.ns
	n = 1

	# fill data vector with pointers to data location
	dataCell = Array{SeisIO.SeisCon}(nsrc)
	for j=1:nsrc
		dataCell[j] = split(data,j)
	end

	return joData{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from cell array of containers
function joData(data::Array{SeisIO.SeisCon,1}; segy_depth_key="RecGroupElevation", vDT::DataType=Float32)
	vDT == Float32 || throw(joDataException("Domain and range types not supported"))

	# length of data vector
	nsrc = length(data)
	numTraces = 0
	for j=1:nsrc
		numTraces += Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data.ns*4))
	end
	m = numTraces*data.ns
	n = 1

	# extract geometry from data container
	geometry = Geometry(data; key="receiver", segy_depth_key=segy_depth_key)

	# fill data vector with pointers to data location
	dataCell = Array{SeisIO.SeisCon}(nsrc)
	for j=1:nsrc
		dataCell[j] = data[j]
	end

	return joData{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from cell array of containers and given geometry
function joData(geometry::Geometry, data::Array{SeisIO.SeisCon}; vDT::DataType=Float32)
	vDT == Float32 || throw(joDataException("Domain and range types not supported"))

	# length of data vector
	nsrc = length(data)
	numTraces = 0
	for j=1:nsrc
		numTraces += Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data[j].ns*4))
	end
	m = numTraces*data[1].ns
	n = 1

	# fill data vector with pointers to data location
	dataCell = Array{SeisIO.SeisCon}(nsrc)
	for j=1:nsrc
		dataCell[j] = data[j]
	end

	return joData{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end



############################################################
## overloaded Base functions

# conj(jo)
conj{vDT}(a::joData{vDT}) =
	joData{vDT}("conj("*a.name*")",a.m,a.n,a.nsrc,a.geometry,a.data)

# transpose(jo)
transpose{vDT}(a::joData{vDT}) =
	joData{vDT}(""*a.name*".'",a.n,a.m,a.nsrc,a.geometry,a.data)
   
# ctranspose(jo)
ctranspose{vDT}(a::joData{vDT}) =
	joData{vDT}(""*a.name*"'",a.n,a.m,a.nsrc,a.geometry,a.data)

##########################################################


# +(joData, joData)
function +{avDT,bvDT}(a::joData{avDT}, b::joData{bvDT})
	size(a) == size(b) || throw(joDataException("dimension mismatch"))
	compareGeometry(a.geometry, b.geometry) == 1 || throw(joDataException("geometry mismatch"))
	c = deepcopy(a)
	c.data = a.data + b.data
	return c
end

# -(joData, joData)
function -{avDT,bvDT}(a::joData{avDT}, b::joData{bvDT})
	size(a) == size(b) || throw(joDataException("dimension mismatch"))
	compareGeometry(a.geometry, b.geometry) == 1 || throw(joDataException("geometry mismatch"))
	c = deepcopy(a)
	c.data = a.data - b.data
	return c
end

# *(joData, number)
function *{avDT}(a::joData{avDT},b::Number)
	c = deepcopy(a)
	c.data = c.data*b
	return c
end

# *(number, joData)
function *{bvDT}(a::Number,b::joData{bvDT})
	c = deepcopy(b)
	c.data = a*c.data
	return c
end

# *(joLinearFunction, joData)
function *{ADDT,ARDT,avDT}(A::joLinearFunction{ADDT,ARDT},v::joData{avDT})
	A.n == size(v,1) || throw(joDataException("shape mismatch"))
	jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,joData):",A.name,typeof(A),avDT]," / "))
	V = A.fop(v)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,joData):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(joLinearOperator, joData)
function *{ADDT,ARDT,avDT}(A::joLinearOperator{ADDT,ARDT},v::joData{avDT})
	A.n == size(v,1) || throw(joDataException("shape mismatch"))
	jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,joData):",A.name,typeof(A),avDT]," / "))
	V = A.fop(v)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,joData):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# /(joData, number)
function -{avDT}(a::joData{avDT},b::Number)
	c = deepcopy(a)
	c.data = c.data/b
	return c
end

# /(number, joData)
function -{bvDT}(a::Number,b::joData{bvDT})
	c = deepcopy(b)
	c.data = a./c.data
	return c
end

# vcat
function vcat{avDT,bvDT}(a::joData{avDT},b::joData{bvDT})
	typeof(a.geometry) == typeof(b.geometry) || throw(joDataException("Geometry type mismatch"))
	m = a.m + b.m
	n = 1
	nsrc = a.nsrc + b.nsrc

	if typeof(a.data) == Array{SeisIO.SeisCon,1} && typeof(b.data) == Array{SeisIO.SeisCon,1}
		data = Array{SeisIO.SeisCon}(nsrc)
	else
		data = Array{Array}(nsrc)
	end

	dt = Array{Any}(nsrc)
	nt = Array{Any}(nsrc)
	t = Array{Any}(nsrc)
	if typeof(data) == Array{SeisIO.SeisCon,1}
		nsamples = Array{Any}(nsrc)
	else
		xloc = Array{Any}(nsrc)
		yloc = Array{Any}(nsrc)
		zloc = Array{Any}(nsrc)
	end

	# Merge data sets and geometries
	for j=1:a.nsrc
		data[j] = a.data[j]
		if typeof(data) == Array{SeisIO.SeisCon,1}
			nsamples[j] = a.geometry.nsamples[j]
		else
			xloc[j] = a.geometry.xloc[j]
			yloc[j] = a.geometry.yloc[j]		
			zloc[j] = a.geometry.zloc[j]
		end
		dt[j] = a.geometry.dt[j]
		nt[j] = a.geometry.nt[j]
		t[j] = a.geometry.t[j]
	end
	for j=a.nsrc+1:nsrc
		data[j] = b.data[j-a.nsrc]
		if typeof(data) == Array{SeisIO.SeisCon,1}
			nsamples[j] = b.geometry.nsamples[j-a.nsrc]
		else
			xloc[j] = b.geometry.xloc[j-a.nsrc]
			yloc[j] = b.geometry.yloc[j-a.nsrc]
			zloc[j] = b.geometry.zloc[j-a.nsrc]
		end
		dt[j] = b.geometry.dt[j-a.nsrc]
		nt[j] = b.geometry.nt[j-a.nsrc]
		t[j] = b.geometry.t[j-a.nsrc]
	end

	if typeof(data) == Array{SeisIO.SeisCon,1}
		geometry = GeometryOOC(data,dt,nt,t,nsamples,a.geometry.key,a.geometry.segy_depth_key)
	else
		geometry = Geometry(xloc,yloc,zloc,dt,nt,t)
	end
	nvDT = promote_type(avDT,bvDT)
	return joData{nvDT}(a.name,m,n,nsrc,geometry,data)
end


# dot product
function dot{avDT,bvDT}(a::joData{avDT}, b::joData{bvDT})
# Dot product for data containers
	size(a) == size(b) || throw(joDataException("dimension mismatch"))
	compareGeometry(a.geometry, b.geometry) == 1 || throw(joDataException("geometry mismatch"))
	dotprod = 0f0
	for j=1:a.nsrc
		dotprod += dot(vec(a.data[j]),vec(b.data[j]))
	end
	return dotprod
end

# norm
function norm{avDT}(a::joData{avDT}; p=2)
	x = 0.f0
	for j=1:a.nsrc
		x += norm(vec(a.data[j]),p)^p
	end
	return x^(1.f0/p)
end

# abs
function abs{avDT}(a::joData{avDT})
	b = deepcopy(a)
	for j=1:a.nsrc
		b.data[j] = abs(a.data[j])
	end
	return b
end

# Subsample data container
"""
    subsample(x,source_numbers)

Subsample seismic data vectors or matrix-free linear operators and extract the entries that correspond\\
to the shot positions defined by `source_numbers`. Works for inputs of type `joData`, `joModeling`, \\
`joProjection`, `joJacobian`, `Geometry`, `joRHS`, `joPDE`, `joPDEfull`.

Examples
========

(1) Extract 2 shots from `joData` vector:

    dsub = subsample(dobs,[1,2])

(2) Extract geometry for shot location 100:

    geometry_sub = subsample(dobs.geometry,100)

(3) Extract Jacobian for shots 10 and 20:

    Jsub = subsample(J,[10,20])

"""
function subsample{avDT}(a::joData{avDT},srcnum)
	geometry = subsample(a.geometry,srcnum)		# Geometry of subsampled data container
	return joData(geometry,a.data[srcnum];vDT=avDT)
end

getindex(x::joData, a) = subsample(x,a)

# Create SeisBlock from joData container to write to file
function joData_to_SeisBlock{avDT}(d::joData{avDT},q::joData{avDT}; source_depth_key="SourceSurfaceElevation", receiver_depth_key="RecGroupElevation")

	typeof(d.geometry) == GeometryOOC && (d.geometry = Geometry(d.geometry))
	typeof(q.geometry) == GeometryOOC && (q.geometry = Geometry(q.geometry))

	blocks = Array{Any}(d.nsrc)
	count = 0
	for j=1:d.nsrc

		# create SeisBlock
		blocks[j] = SeisBlock(d.data[j])
		numTraces = size(d.data[j],2)
		traceNumbers = convert(Array{Integer,1},count+1:count+numTraces)

		# set headers
		set_header!(blocks[j], "GroupX", convert(Array{Integer,1},round.(d.geometry.xloc[j]*1f3)))
		if length(d.geometry.yloc[j]) == 1
			set_header!(blocks[j], "GroupY", Int(round(d.geometry.yloc[j]*1f3)))
		else
			set_header!(blocks[j], "GroupY", convert(Array{Integer,1},round.(d.geometry.yloc[j]*1f3)))
		end
		set_header!(blocks[j], receiver_depth_key, convert(Array{Integer,1},round.(d.geometry.zloc[j]*1f3)))
		set_header!(blocks[j], "SourceX", Int(round.(q.geometry.xloc[j]*1f3)))
		set_header!(blocks[j], "SourceY", Int(round.(q.geometry.yloc[j]*1f3)))
		set_header!(blocks[j], source_depth_key, Int(round.(q.geometry.zloc[j]*1f3)))

		set_header!(blocks[j], "dt", Int(d.geometry.dt[j]*1f3))
		set_header!(blocks[j], "FieldRecord",j)
		set_header!(blocks[j], "TraceNumWithinLine", traceNumbers)
		set_header!(blocks[j], "TraceNumWithinFile", traceNumbers)
		set_header!(blocks[j], "TraceNumber", traceNumbers)
		set_header!(blocks[j], "ElevationScalar", -1000)
		set_header!(blocks[j], "RecSourceScalar", -1000)
		count += numTraces
	end

	# merge into single block
	fullblock = blocks[1]
	for j=2:d.nsrc
		fullblock = merge(fullblock,blocks[j])
		blocks[j] = []
	end
	return fullblock
end

###########################################################################################################

# Overload base function for SeisIO objects

vec(x::SeisIO.SeisCon) = vec(x[1].data)
norm(x::SeisIO.IBMFloat32; kwargs...) = norm(convert(Float32,x); kwargs...)
dot(x::SeisIO.IBMFloat32, y::SeisIO.IBMFloat32) = dot(convert(Float32,x), convert(Float32,y))
dot(x::SeisIO.IBMFloat32, y::Float32) = dot(convert(Float32,x), y)
dot(x::Float32, y::SeisIO.IBMFloat32) = dot(x, convert(Float32,y))

# binary operations return dense arrays
+(x::SeisIO.SeisCon, y::SeisIO.SeisCon) = +(x[1].data,y[1].data)
+(x::SeisIO.IBMFloat32, y::SeisIO.IBMFloat32) = +(convert(Float32,x),convert(Float32,y))
+(x::SeisIO.IBMFloat32, y::Float32) = +(convert(Float32,x),y)
+(x::Float32, y::SeisIO.IBMFloat32) = +(x,convert(Float32,y))

-(x::SeisIO.SeisCon, y::SeisIO.SeisCon) = -(x[1].data,y[1].data)
-(x::SeisIO.IBMFloat32, y::SeisIO.IBMFloat32) = -(convert(Float32,x),convert(Float32,y))
-(x::SeisIO.IBMFloat32, y::Float32) = -(convert(Float32,x),y)
-(x::Float32, y::SeisIO.IBMFloat32) = -(x,convert(Float32,y))

+(a::Number, x::SeisIO.SeisCon) = +(a,x[1].data)
+(x::SeisIO.SeisCon, a::Number) = +(x[1].data,a)

-(a::Number, x::SeisIO.SeisCon) = -(a,x[1].data)
-(x::SeisIO.SeisCon, a::Number) = -(x[1].data,a)

*(a::Number, x::SeisIO.SeisCon) = *(a,x[1].data)
*(x::SeisIO.SeisCon, a::Number) = *(x[1].data,a)

/(a::Number, x::SeisIO.SeisCon) = /(a,x[1].data)
/(x::SeisIO.SeisCon, a::Number) = /(x[1].data,a)


