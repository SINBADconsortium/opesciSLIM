# 2D FWI on Overthrust model using minConf library 
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
v0 = sqrt.(1./model0.m)
vmin = ones(Float32,model0.n) + 0.3f0
vmax = ones(Float32,model0.n) + 5.5f0
vmin[:,1:21] = v0[:,1:21]	# keep water column fixed
vmax[:,1:21] = v0[:,1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0./vmax).^2)
mmax = vec((1f0./vmin).^2)

# Load data
block = segy_read("/scratch/slim/pwitte/overthrust2D/overthrust_2D.segy")
dobs = joData(block)

# Set up wavelet
src_geometry = Geometry(block; key="source", segy_depth_key="SourceDepth")
wavelet = source(src_geometry.t[1],src_geometry.dt[1],0.008f0)	# 8 Hz wavelet
q = joData(src_geometry,wavelet)

############################### FWI ###########################################

# Optimization parameters
srand(1)	# set seed of random number generator
niterations = 20
batchsize = 20
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)],2),model0.n)	# bound projection operator
fTerm = 1f-2
gradTerm = 1f-4
num_feval = 0
max_feval = 100
fval = 0
fvals = zeros(Float32,niterations)
dsub = []
qsub = []

# evaluate FWI misfit function for current model
function misfit(x)	
	global num_feval; num_feval += 1
	model_new = deepcopy(model0)
	model_new.m = proj(reshape(x,model0.n))
	return f,g = fwi_objective(model_new,qsub,dsub)
end

# Backtracking line search with Armijo-Goldstein condition
function backtracking_linesearch(objective, x, grad; f_curr=[], alpha=1f0, tau=.5f0, c=.5f0, maxiter=10)
	
	# Parameters
	p = -grad/norm(grad)	# normalized descent direction
	m = dot(p,grad)
	t = -c*m
	isempty(f_curr) && (f_curr = objective(x)[1])	# current function value
	f_new = objective(x + alpha*p)[1]	# function value with initial step size
	iteration = 0

	# Line search
	while(f_curr - f_new < alpha*t && iteration < maxiter)
		println("	Iter LS: ", iteration, "; ",f_curr - f_new," >= ", alpha*t, "; alpha: ",alpha)
		alpha *= tau
		f_new = objective(x + alpha*p)[1]
		iteration += 1
	end
	return alpha*p
end

# Line search with armijo rule and curvature
function linesearch_wolfe(objective, x, grad_curr; f_curr=[], alpha=1f-4, tau=10f0, c1=1f-4, c2=.1f0, maxiter=10)

	# Parameters
	p = -grad_curr/norm(grad_curr)	# normalized descent direction
	isempty(f_curr) && ((f_curr, grad_curr) = objective(x))	# current function value and gradient
	f_new, grad_new = objective(x + alpha*p)	# function value and gradient with initial step size
	iteration = 0

	# line search with wolfe conditions
	while(f_new > f_curr + c1*alpha*dot(p,grad_curr) || -dot(p,grad_new) > -c2*dot(p,grad_curr) && iteration < maxiter)

		println("	Iter LS: ", iteration, "; ",f_new," <= ", f_curr + c1*alpha*dot(p,grad_curr),"; ",
		        -dot(p,grad_new), " <= ", -c2*dot(p,grad_curr), "; alpha: ",alpha)
		alpha *= 5f0
		f_new, grad_new = objective(x + alpha*p)
		iteration += 1
	end
	return alpha*p
end

# Main loop
for j=1:niterations

	# select current batch
	idx = randperm(dobs.nsrc)[1:batchsize]
	dsub = subsample(dobs,idx)
	qsub = subsample(q,idx)

	# get fwi objective function value and gradient
	fval, gradient = fwi_objective(model0,qsub,dsub)
	println("FWI iteration no: ",j,"; function value: ",fval)

	# linesearch
	step = backtracking_linesearch(misfit, vec(model0.m), gradient; f_curr=fval, alpha=10.f0, tau=.1f0, c=1e-4)
	#step = linesearch_wolfe(misfit, vec(model0.m), gradient; f_curr=fval, alpha=1f-4, tau=5f0, c1=1f-4, c2=.9f0)
	
	# Update model and bound projection
	model0.m = proj(model0.m + reshape(step,model0.n))

	# termination criteria
	fvals[j] = fval
	#if fval <= fTerm || norm(gradient) <= gradTerm || num_feval > max_feval
	#	break
	#end

end

# Save results
write(h5open("results_linesearch_armijo.h5", "w"), "x", sqrt.(1./reshape(x,model0.n)), "fsave", fsave, "fhistory",fvals)
# write(h5open("results_linesearch_wolfe.h5", "w"), "x", sqrt.(1./reshape(x,model0.n)), "fsave", fsave, "fhistory",fvals)


