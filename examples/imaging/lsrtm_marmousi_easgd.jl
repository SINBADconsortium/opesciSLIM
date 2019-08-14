# LSRTM of Marmousi using stochastic gradient descent
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: January 2017

using PyCall,PyPlot,HDF5,opesciSLIM.TimeModeling,opesciSLIM.LeastSquaresMigration, SeisIO

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

# Elastic average stochastic gradient descent
niter = 32
p = 10
batchsize = 1

eta = 0.03
rho = 1.
alpha = eta*rho
beta = p*alpha

x = zeros(Float32,info.n,p)
xnew = zeros(Float32,info.n,p)
xav = zeros(Float32,info.n)

# Parallel gradient function
@everywhere function update_x(Ml,J,Mr,x,d,eta,alpha,xav)

	# gradient
	r = Ml*J*Mr*x - Ml*d
	g = Mr'*J'*Ml'*r

	# Update variable
	return x - eta*g - alpha*(x - xav)
end
update_x_par = remote(update_x)		# parallel function instance

# Main loop
figure()
for j=1:niter
	println("Iteration: ", j)

	@sync begin
		for k=1:p

			# Select batch
			idx = randperm(dD.nsrc)[1:batchsize]
			Jsub = subsample(J,idx)
			dsub = subsample(dD,idx)
			Ml = opMarineTopmute(30,dsub.geometry)	# data topmute

			# Calculate x update
			@async xnew[:,k] = update_x_par(Ml,Jsub,Mr,x[:,k],dsub,eta,alpha,xav)
		end
	end

	xav = (1-beta)*xav + beta*(1/p *sum(x,2))
	x = copy(xnew)

end

# Save final velocity model, function value and history
h5open("lsrtm_marmousi_easgd_result.h5", "w") do file
	write(file, "x", reshape(x,model0.n), "fval", fval)
end



