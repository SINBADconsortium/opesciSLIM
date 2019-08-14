using PyCall, PyPlot, HDF5, opesciSLIM.TimeModeling, opesciSLIM.SLIM_optim, SeisIO

# Load overthrust model
~isfile("overthrust_3D_initial_model.h5") && run(`wget ftp://ftp.slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5`)
n,d,o,m = read(h5open("overthrust_3D_initial_model.h5","r"), "n", "d", "o", "m")

# Set up model structure
model = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m)

# Bound constraints
vmin = ones(Float32,model0.n) + 0.4f0
vmax = ones(Float32,model0.n) + 5.5f0
mmin = vec((1f0./vmax).^2); vmax=[]
mmax = vec((1f0./vmin).^2); vmin=[]

# Load data container containing filenames
container = segy_scan("/path/to/shot/records/","overthrust_3D_shot",["GroupX","GroupY","RecGroupElevation","SourceSurfaceElevation","dt"])
dobs = joData(container)

# Set up source
src_geometry = Geometry(container; key="source")
wavelet = source(src_geometry.t[1], src_geometry.dt[1], 0.008)	# 8 Hz peak frequency
q = joData(src_geometry, wavelet)

############################### FWI ###########################################

# Optimization parameters
srand(1)	# set seed of random number generator
fevals = 15
batchsize = 540
count = 0
fvals = zeros(50)
opt = Options(limit_m=true,
			  save_rate=14.,		# sampling rate of gradient
			  buffer_size=500.
			  )

function objective_function(x)

	# update model and save snapshot
	model0.m = reshape(x,model0.n);
	write(h5open(join(["snapshot_3D_FWI_iteration_",string(count),".h5"]),"w"),"v",sqrt.(1./model0.m))

	# select batch
	idx = randperm(dobs.nsrc)[1:batchsize]

	# fwi function value and gradient
	fval, grad = fwi_objective(model0,q[idx],dobs[idx];options=opt); gc()
	grad = reshape(grad,model0.n); grad[:,:,1:21] = 0.f0	# reset gradient in water column to 0.
	grad = .125f0*grad/maximum(abs(grad))	# scale gradient to help line search
	global count; count += 1; fvals[count] = fval

    return fval, vec(grad)
end

# Bound projection
ProjBound(x) = median([mmin x mmax],2)

# FWI with SPG
spg_opt = spg_options(verbose=3, maxIter=fevals, memory=1)
x, fsave, funEvals= minConf_SPG(objective_function, vec(model0.m), ProjBound, spg_opt)

# Save results
write(h5open("results_3D_FWI.h5", "w"), "x", sqrt.(1./reshape(x,model0.n)), "fsave", fsave, "fhistory",fvals)


