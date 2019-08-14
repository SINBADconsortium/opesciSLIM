# Example for 2D FWI of Camembert model using SPG
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, opesciSLIM.TimeModeling, NLopt, opesciSLIM.SLIM_optim, HDF5

# Load camembert model
~isfile("camembert.h5") && run(`wget ftp://ftp.slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/camembert.h5`)
n,d,o,m = read(h5open("camembert.h5","r"), "n", "d", "o", "m")

# Set up model structure
model = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m)

# Bound constraints and initial model
m0 = ones(Float32,model.n)*0.25f0
vmin = ones(Float32,model.n) + 0.3f0
vmax = ones(Float32,model.n) + 4f0
mmin = vec((1f0./vmax).^2)
mmax = vec((1f0./vmin).^2)

# Setup info and model structure
nsrc = 10	# number of sources
model0 = Model(model.n,model.d,model.o,m0)

## Set up receiver geometry
nxrec = 100
xrec = ones(Float32,100)*50f0
yrec = 0f0
zrec = linspace(50f0,1950f0,nxrec)

# receiver sampling and recording time
timeR = 3000f0	# receiver recording time [ms]
dtR = 4f0	# receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(linspace(1950f0, 1950f0, nsrc))
ysrc = convertToCell(linspace(0f0, 0f0, nsrc))
zsrc = convertToCell(linspace(50f0, 1950f0, nsrc))

# source sampling and number of time steps
timeS = 3000f0
dtS = 4f0	# receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# setup wavelet
f0 = 0.004f0
wavelet = source(timeS,dtS,f0)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry,recGeometry,model)
info = Info(prod(model.n),nsrc,ntComp)

###################################################################################################

# Combined operator Pr*F*Ps'
F = joModeling(info,model,srcGeometry,recGeometry)
q = joData(srcGeometry,wavelet)

# Nonlinear modeling
dobs = F*q

# Optimization library
opt = "minConf"	# or "NLopt"

###################################################################################################

if opt=="minConf"

	# Optimization parameters
	iterations = 20
	batchsize = 10

	function objective_function(x)
		model0.m = reshape(x,model0.n);

		# select batch
		idx = randperm(dobs.nsrc)[1:batchsize]

		# fwi function value and gradient
		fval,grad = fwi_objective(model0,q[idx],dobs[idx])
		grad = reshape(grad,model0.n); grad[1:20,:] = 0.; grad[end-20:end,:] = 0.;

 	   return fval, vec(grad)
	end

	ProjBound(x) = vec(reshape(median([vec(mmin) vec(x) vec(mmax)],2), model.n))

	# FWI with SPG
	#options = spg_options(verbose=3, maxIter=iterations, memory=3)
	#x, fsave, funEvals= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)

	# FWI with PQN
	options = pqn_options(verbose=3, maxIter=iterations, corrections=15)
	x, fsave, funEvals= minConf_PQN(objective_function, vec(model0.m), ProjBound, options)

####################################################################################################

elseif opt=="NLopt"

	# NLopt objective function
	batchsize = 10;	# number of sources per iteration
	count = 0
	function f!(x,grad)
		# calcualte gradient and function value
		model0.m = convert(Array{Float32,2},reshape(x,model0.n))
		idx = randperm(dobs.nsrc)[1:batchsize]
		fval,gradient = fwi_objective(model0,subsample(q,idx),subsample(dobs,idx))
	
		# reset water column to zero
		gradient = reshape(gradient,model0.n); gradient[1:20,:] = 0f0; gradient[180:end,:] = 0f0
		grad[1:end] = vec(gradient)

		global count; count += 1
		return convert(Float64,fval)
	end

	# Optimization parameters
	opt = Opt(:LD_LBFGS, info.n)
	lower_bounds!(opt, mmin)
	upper_bounds!(opt, mmax)
	maxeval!(opt,30)
	maxtime!(opt,100)
	min_objective!(opt,f!)
	(minf, minx, ret) = optimize(opt, vec(model0.m))

else
	throw("wrong optimization keyword")
end

