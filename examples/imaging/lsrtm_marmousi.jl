# LSRTM of Marmousi using stochastic gradient descent
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: January 2017

using PyCall,PyPlot, HDF5, opesciSLIM.TimeModeling, opesciSLIM.LeastSquaresMigration, SeisIO, JLD

# Fetch data and starting model from ftp server
~isfile("marmousi_2D.segy") && run(`wget ftp://ftp.slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_2D.segy`)
~isfile("marmousi_migration_velocity.h5") && run(`wget ftp://ftp.slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_migration_velocity.h5`)

# Load migration velocity model
n,d,o,m0 = read(h5open("marmousi_migration_velocity.h5","r"), "n", "d", "o", "m0")

# Load data
block = segy_read("marmousi_2D.segy")
dD = joData(block)

# Set up model structure
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Set up wavelet
src_geometry = Geometry(block; key="source", segy_depth_key="SourceDepth")
wavelet = source(src_geometry.t[1],src_geometry.dt[1],0.03)	# 30 Hz wavelet
q = joData(src_geometry,wavelet)

# Set up info structure
ntComp = get_computational_nt(q.geometry,dD.geometry,model0)	# no. of computational time steps
info = Info(prod(model0.n),dD.nsrc,ntComp)


###################################################################################################

# Setup operators
F = joModeling(info,model0,q.geometry,dD.geometry)
J = joJacobian(F,q)

# Right-hand preconditioners (model topmute)
Mr = opTopmute(model0.n,42,10)

# Stochastic gradient
x = zeros(Float32,info.n)
batchsize = 10
niter = 32
fval = zeros(Float32,niter)

# Main loop
for j=1:niter
	println("Iteration: ", j)

	# Select batch and set up left-hand preconditioner
	idx = randperm(dD.nsrc)[1:batchsize]
	Jsub = subsample(J,idx)
	dsub = subsample(dD,idx)
	Ml = opMarineTopmute(30,dsub.geometry)	# data topmute

	# Compute residual and gradient
	r = Ml*Jsub*Mr*x - Ml*dsub
	g = Mr'*Jsub'*Ml'*r

	# Step size and update variable
	fval[j] = .5*norm(r)^2
	t = norm(r)^2/norm(g)^2
	x -= t*g

	save("snapshot.jld","x",reshape(x,model0.n))

end

# Save final velocity model, function value and history
h5open("lsrtm_marmousi_2D_result.h5", "w") do file
	write(file, "x", reshape(x,model0.n), "fval", fval)
end


