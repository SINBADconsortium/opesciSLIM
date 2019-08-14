# Full waveform inversion on the 2D Overthrust model using a spectral projected gradient algorithm
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: May 2017
#

using PyCall, PyPlot, HDF5, opesciSLIM.TimeModeling, opesciSLIM.SLIM_optim, SeisIO

# Load starting model
~isfile("overthrust_2D_initial_model.h5") && run(`wget ftp://ftp.slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D_initial_model.h5`)
n,d,o,m0 = read(h5open("overthrust_2D_initial_model.h5","r"), "n", "d", "o", "m0")

# Set up model structure
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
vmin = ones(Float32,model0.n) + 0.4f0
vmax = ones(Float32,model0.n) + 5.5f0

# Slowness squared [s^2/km^2]
mmin = vec((1f0./vmax).^2)
mmax = vec((1f0./vmin).^2)

# Load data
~isfile("overthrust_2D.segy") && run(`wget ftp://ftp.slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D.segy`)
block = segy_read("overthrust_2D.segy")
dobs = joData(block)

# Set up wavelet
src_geometry = Geometry(block; key="source", segy_depth_key="SourceDepth")
wavelet = source(src_geometry.t[1],src_geometry.dt[1],0.008f0)	# 8 Hz wavelet
q = joData(src_geometry,wavelet)

############################### FWI ###########################################

# Optimization parameters
srand(1)	# set seed of random number generator
fevals = 20
batchsize = 20
fvals = zeros(25)

# Set up modeling options
opt = Options(save_rate=14f0)

# Objective function for library
count = 0
function objective_function(x)
	model0.m = reshape(x,model0.n);

	# select batch
	idx = randperm(dobs.nsrc)[1:batchsize]

	# fwi function value and gradient
	fval,grad = fwi_objective(model0,q[idx],dobs[idx];options=opt)
	grad = reshape(grad,model0.n); grad[:,1:21] = 0.f0	# reset gradient in water column to 0.
	grad = .125f0*grad/maximum(abs.(grad))
	
	global count; count+= 1; fvals[count] = fval
    return fval, vec(grad)
end

# Bound projection
ProjBound(x) = median([mmin x mmax],2)

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=3)
x, fsave, funEvals= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)

# Save results
write(h5open("results.h5", "w"), "x", sqrt.(1./reshape(x,model0.n)), "fsave", fsave, "fhistory",fvals)

# Plots
figure(); imshow(sqrt.(1./reshape(m0,model0.n))',cmap="jet"); title("Initial model")
figure(); imshow(sqrt.(1./x)',cmap="jet"); title("FWI")
figure(); plot(fvals); title("Function value")


