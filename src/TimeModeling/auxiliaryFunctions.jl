# Auxiliary functions for TimeModeling module
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: September 2016
#

export source, add_pml, remove_pml, get_computational_nt, smooth10, damp_boundary, calculate_dt, setup_grid, setup_3D_grid
export convertToCell, reshape_for_devito, reshape_from_devito, limit_model_to_receiver_area, extend_gradient, plot_geometry_map


function limit_model_to_receiver_area(srcGeometry::Geometry,recGeometry::Geometry,model::Model,buffer;pert=[])
	# Restrict full velocity model to area that contains either sources and receivers

	# scan for minimum and maximum x and y source/receiver coordinates
	min_x = minimum([vec(recGeometry.xloc[1]); vec(srcGeometry.xloc[1])])
	max_x = maximum([vec(recGeometry.xloc[1]); vec(srcGeometry.xloc[1])])
	min_y = minimum([vec(recGeometry.yloc[1]); vec(srcGeometry.yloc[1])])
	max_y = maximum([vec(recGeometry.yloc[1]); vec(srcGeometry.yloc[1])])

	# add buffer zone if possible
	min_x = max(model.o[1],min_x-buffer)
	max_x = min(model.o[1] + model.d[1]*(model.n[1]-1),max_x+buffer)
	min_y = max(model.o[2],min_y-buffer)
	max_y = min(model.o[2] + model.d[2]*(model.n[2]-1),max_y+buffer)

	# extract part of the model that contains sources/receivers
	nx_min = Int(round(min_x/model.d[1])) + 1
	nx_max = Int(round(max_x/model.d[1])) + 1
	ny_min = Int(round(min_y/model.d[2])) + 1
	ny_max = Int(round(max_y/model.d[2])) + 1
	ox = (nx_min - 1)*model.d[1]
	oy = (ny_min - 1)*model.d[2]
	oz = model.o[3]

	# Extract relevant model part from full domain
	n_orig = model.n
	model.m = model.m[nx_min:nx_max,ny_min:ny_max,:]
	model.o = (ox,oy,oz)
	model.n = size(model.m)
	if isempty(pert)
		return model
	else
		pert = reshape(pert,n_orig)[nx_min:nx_max,ny_min:ny_max,:]
		return model,vec(pert)
	end
end

function extend_gradient(model_full::Model,model::Model,gradient::Array)
	# Extend gradient back to full model size
	full_gradient = zeros(Float32,model_full.n)
	nx_start = Int((model.o[1] - model_full.o[1])/model.d[1] + 1)
	nx_end = nx_start + model.n[1] - 1
	ny_start = Int((model.o[2] - model_full.o[2])/model.d[2] + 1)
	ny_end = ny_start + model.n[2] - 1 
	full_gradient[nx_start:nx_end,ny_start:ny_end,:] = gradient
	return full_gradient
end

function ccallsyms(ccallargs, n, argsyms)
	# create argument list for ccall
    if n > 0
        if length(argsyms) == n
            ccallargs = Any[ccallargs..., argsyms...]
        else
            for i = 1:length(argsyms)-1
                push!(ccallargs, argsyms[i])
            end
            for i = 1:n-length(argsyms)+1
                push!(ccallargs, Expr(:getindex, argsyms[end], i))
            end
        end
    end
    ccallargs
end

function ccallexpr(lib::String, ccallsym::Symbol, argtypes::Array{DataType,1}, argsyms::Array{Any,1})
    ccallargs = Any[Libdl.dlsym(Libdl.dlopen(lib),ccallsym), Cint, Expr(:tuple, argtypes...)]
    ccallargs = ccallsyms(ccallargs, length(argtypes), argsyms)
    eval(:(ccall($(ccallargs...))))
end

"""
Processing of input varibales for the ccall from the symbolic values obtained from devito
Some convention is applied here for the keywords
forward wavefield : u
adjoint wavefield : v
square slowness : m
boundary damping : damp
density : rho
source : src
source position : src_coords
data : rec
data locations : rec_coords
model has to be an input t infer the number of dimensions and size of the domain
use as :

op = devito[:operator]()
args = process_args(op, model; u=u, rec=...)
"""
function process_args(op, kwargs_dict)
	# kwargs_dict = Dict(kwargs)
	args = op[:arguments]()
	nargs = length(args)
	TypesIn = []
	ArgsIn = Array{Any}(length(args))
	for i=1:length(args)
		if args[i][2] == nothing
			loc = string(args[i][1])
			TypesIn = [TypesIn..., Ref{Cfloat}]
			if haskey(kwargs_dict, Symbol(args[i][1]))
				ArgsIn[i] = kwargs_dict[Symbol(args[i][1])]
			else
				@printf("Unrecognized key %s", Symbol(args[i][1]))
				error("Unknown key for inputs, check the inputs")
			end
		else
			TypesIn = [TypesIn..., Cint]
			ArgsIn[i] = args[i][2]
		end
	end
	return TypesIn, ArgsIn
end

function add_pml(array,nb)
	# extent model domain for absorbing boundaries
	if length(array)>1
		pad_list =[(nb, nb),(nb, nb)]	# x,z
		if ndims(array) == 3
			pad_list =[(nb, nb), (nb, nb), (nb, nb)]	# x,y,z
		end
		return np.pad(array, pad_list, "edge")
	else
		return array
	end
end

function remove_pml(array,nb)
# Cut off PML from model with Python dimensions
    if ndims(array) == 3
       return array[nb+1:end-nb, nb+1:end-nb, nb+1:end-nb]	# x,y,z
    else 
       return array[nb+1:end-nb, nb+1:end-nb]	# x,z
    end
end

function damp_boundary(damp, model)
	# add absorbing boundaries to model
	
	# Number of dimensions
    num_dim = ndims(damp)
    
	# Damping coefficients
	if	num_dim == 2
		dampcoeffX = 1.5 * log(1.0 / 0.001) / (model.nb * model.d[1])
    	dampcoeffZ = 1.5 * log(1.0 / 0.001) / (model.nb * model.d[2])
	else
		dampcoeffX = 1.5 * log(1.0 / 0.001) / (model.nb * model.d[1])
		dampcoeffY = 1.5 * log(1.0 / 0.001) / (model.nb * model.d[2])
    	dampcoeffZ = 1.5 * log(1.0 / 0.001) / (model.nb * model.d[3])
	end

	if num_dim == 2
		for i=1:model.nb	# add x pml
        	pos = abs((model.nb-i)/float(model.nb))
        	val = dampcoeffX * (pos - sin(2*pi*pos)/(2*pi))
        	damp[i, :] += val
        	damp[end-i+1, :] += val
		end
    	for i=1:model.nb	# add z pml
       		pos = abs((model.nb-i)/float(model.nb))
       		val = dampcoeffZ * (pos - sin(2*pi*pos)/(2*pi))
        	damp[:, i] += val
        	damp[:, end-i+1] += val
		end
	else
		for i=1:model.nb	# add x pml
        	pos = abs((model.nb-i)/float(model.nb))
        	val = dampcoeffX * (pos - sin(2*pi*pos)/(2*pi))
            damp[i, :, :] += val
            damp[end-i+1, :, :] += val
		end
		for i=1:model.nb	# add y pml
        	pos = abs((model.nb-i)/float(model.nb))
        	val = dampcoeffY * (pos - sin(2*pi*pos)/(2*pi))
            damp[:, i, :] += val
            damp[:, end-i+1, :] += val
		end
	   	for i=1:model.nb	# add z pml
       		pos = abs((model.nb-i)/float(model.nb))
       		val = dampcoeffZ * (pos - sin(2*pi*pos)/(2*pi))
            damp[:, :, i] += val
            damp[:, :, end-i+1] += val
		end
	end
	return damp
end

"""
    convertToCell(x)

Convert an array `x` to a cell array (`Array{Any,1}`) with `length(x)` entries,\\
where the i-th cell contains the i-th entry of `x`.

"""
function convertToCell(x)
	n = length(x)
	y = Array{Any}(n)
	for j=1:n
		y[j] = x[j]
	end
	return y
end

# 1D source time function
"""
    source(tmax, dt, f0)

Create seismic Ricker wavelet of length `tmax` (in milliseconds) with sampling interval `dt` (in milliseonds)\\
and central frequency `f0` (in kHz).

"""
function source(tmax, dt, f0)
	t0 = 0.
	nt = Int(trunc((tmax - t0)/dt + 1))
	t = linspace(t0,tmax,nt)
	r = (pi * f0 * (t - 1 / f0))
	q = zeros(Float32,nt,1)
	q[:,1] = (1. - 2.*r.^2.).*exp.(-r.^2.)
	return vec(q)
end

function calculate_dt(n,d,o,v; epsilon=0)
	model = ct.IGrid(o, d, n, v, epsilon=epsilon)
	return dt = model[:critical_dt]
end

"""
    get_computational_nt(srcGeometry, recGeoemtry, model)

Estimate the number of computational time steps. Required for calculating the dimensions\\
of the matrix-free linear modeling operators. `srcGeometry` and `recGeometry` are source\\
and receiver geometries of type `Geometry` and `model` is the model structure of type \\
`Model`.

"""
function get_computational_nt(srcGeometry, recGeometry, model::ModelTTI)
	# Determine number of computational time steps
	if typeof(srcGeometry) == GeometryOOC
		nsrc = length(srcGeometry.container)
	else
		nsrc = length(srcGeometry.xloc)
	end
	nt = Array{Any}(nsrc)
	dtComp = calculate_dt(model.n,model.d,model.o,sqrt.(1./model.m); epsilon=model.epsilon)
	for j=1:nsrc
		ntRec = Int(trunc(recGeometry.dt[j]*(recGeometry.nt[j]-1))) / dtComp
		ntSrc = Int(trunc(srcGeometry.dt[j]*(srcGeometry.nt[j]-1))) / dtComp
		nt[j] = max(Int(trunc(ntRec)), Int(trunc(ntSrc)))
	end
	return nt
end

function get_computational_nt(srcGeometry, recGeometry, model::Union{Model,ModelRho})
	# Determine number of computational time steps
	if typeof(srcGeometry) == GeometryOOC
		nsrc = length(srcGeometry.container)
	else
		nsrc = length(srcGeometry.xloc)
	end
	nt = Array{Any}(nsrc)
	dtComp = calculate_dt(model.n,model.d,model.o,sqrt.(1./model.m))
	for j=1:nsrc
		ntRec = Int(trunc(recGeometry.dt[j]*(recGeometry.nt[j]-1))) / dtComp
		ntSrc = Int(trunc(srcGeometry.dt[j]*(srcGeometry.nt[j]-1))) / dtComp
		nt[j] = max(Int(trunc(ntRec)), Int(trunc(ntSrc)))
	end
	return nt
end

function setup_grid(geometry,n, origin)
	# 3D grid 
	if length(n)==3
		if length(geometry.xloc[1]) > 1
			source_coords = Array{Float32,2}([vec(geometry.xloc[1]) vec(geometry.yloc[1]) vec(geometry.zloc[1])])
		else
			source_coords = Array{Float32,2}([geometry.xloc[1] geometry.yloc[1] geometry.zloc[1]])
		end
		orig = Array{Float32}([origin[1] origin[2] origin[3]])
	else
	# 2D grid
		if length(geometry.xloc[1]) > 1
			source_coords = Array{Float32,2}([vec(geometry.xloc[1]) vec(geometry.zloc[1])])
		else
			source_coords = Array{Float32,2}([geometry.xloc[1] geometry.zloc[1]])
		end
		orig = Array{Float32}([origin[1] origin[2]])
	end
	return source_coords .- orig
end

function setup_3D_grid(xrec::Array{Any,1},yrec::Array{Any,1},zrec::Array{Any,1})
	# Take input 1d x and y coordinate vectors and generate 3d grid. Input are cell arrays
	nsrc = length(xrec)
	xloc = Array{Any}(nsrc)
	yloc = Array{Any}(nsrc)
	zloc = Array{Any}(nsrc)
	for i=1:nsrc
		nxrec = length(xrec[i])
		nyrec = length(yrec[i])
	
		xloc[i] = zeros(nxrec*nyrec)
		yloc[i] = zeros(nxrec*nyrec)
		zloc[i] = zeros(nxrec*nyrec)
	
		idx = 1

		for k=1:nyrec
			for j=1:nxrec
				xloc[i][idx] = xrec[i][j]
				yloc[i][idx] = yrec[i][k]
				zloc[i][idx] = zrec[i]
				idx += 1
			end
		end
	end
	return xloc, yloc, zloc
end

function setup_3D_grid(xrec,yrec,zrec)
# Take input 1d x and y coordinate vectors and generate 3d grid. Input are arrays/linspace
	nxrec = length(xrec)
	nyrec = length(yrec)

	xloc = zeros(nxrec*nyrec)
	yloc = zeros(nxrec*nyrec)
	zloc = zeros(nxrec*nyrec)
	idx = 1
	for k=1:nyrec
		for j=1:nxrec
			xloc[idx] = xrec[j]
			yloc[idx] = yrec[k]
			zloc[idx] = zrec
			idx += 1
		end
	end
	return xloc, yloc, zloc
end

function smooth10(velocity,shape)
	# 10 point smoothing function
	out = ones(Float32,shape)
	nz = shape[end]
	if length(shape)==3
		out[:,:,:] = velocity[:,:,:]
		for a=5:nz-6
			out[:,:,a] = sum(velocity[:,:,a-4:a+5],3) / 10
		end
	else
		out[:,:] = velocity[:,:]
		for a=5:nz-6
			out[:,a] = sum(velocity[:,a-4:a+5],2) / 10
		end
	end
	return out
end

# Vectorization of single variable (not defined in Julia)
vec(x::Float64) = x;
vec(x::Float32) = x;
vec(x::Int64) = x;
vec(x::Int32) = x;

function plot_geometry_map(model,srcGeometry,recGeometry,shot_no;colormap="viridis")
	close("all")
	map = zeros(Float32,model.n[1],model.n[2])

	xmin = minimum(vec(recGeometry.xloc[shot_no]))
	xmax = maximum(vec(recGeometry.xloc[shot_no]))
	ymin = minimum(vec(recGeometry.yloc[shot_no]))
	ymax = maximum(vec(recGeometry.yloc[shot_no]))

	xmin_n = Int(round((xmin-model.o[1])/model.d[1] + 1))
	xmax_n = Int(round((xmax-model.o[1])/model.d[1] + 1))
	ymin_n = Int(round((ymin-model.o[2])/model.d[2] + 1))
	ymax_n = Int(round((ymax-model.o[2])/model.d[2] + 1))

	src_x = Int(round((srcGeometry.xloc[shot_no][1]-model.o[1])/model.d[1] + 1))
	src_y = Int(round((srcGeometry.yloc[shot_no][1]-model.o[2])/model.d[2] + 1))

	map[xmin_n:xmax_n,ymin_n:ymax_n] = 1.
	#map[src_x,src_y] = -1.
	imshow(map; cmap=colormap); colorbar()
	xlabel("y-coordinate")
	ylabel("x-coordinate")
	plot(src_y,src_x,"x")
end

wrap_retry(f,n) = retry(f;n=n)


