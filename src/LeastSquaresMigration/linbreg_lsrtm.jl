# Sparsity-promoting least squares migration with source subsampling using the linearized Bregman algorithm
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: December 2016
#

export linbreg_lsrtm

function linbreg_lsrtm(J::TimeModeling.joJacobian, dD::TimeModeling.joData; iterations=10, lambda=[], shots_per_iteration=1, snapshot=[])

# Sparsity promoting LSRTM using the linearized Bregman method

	# Setup Curvelet transform
	C = joCurvelet2D(J.model.n[1], J.model.n[2]; zero_finest=true, DDT=Float32, RDT=Float64)

	# Main loop
	x = zeros(Float32,prod(J.model.n))
	z = zeros(Float32,prod(J.model.n))
	for j=1:iterations
		println("Iteration ", j, " of ", iterations)

		# Subsample shots
		shotIdx = randperm(J.info.nsrc)[1:shots_per_iteration]

		# Subsample observed data
		dDsub = subsample(dD, shotIdx)
		
		# Subsample Jacobian
		Jsub = subsample(J, shotIdx)

		# Residual
		dDpred = Jsub*x
		residual = dDpred - dDsub

		# Gradient
		gradient = Jsub'*residual

		# Step size
		step = norm(residual)^2/norm(gradient)^2
		
		# Automatic thresholding
		if isempty(lambda) && j==1
			lambda = convert(Float32,0.1*maximum(abs.(step*C*gradient)))
		end
		
		# Update dual variable
		z = z - step*gradient;

		# Update primal variable
		x = C'*softThresholding(C*z, lambda)

		# Save snapshots
		if ~isempty(snapshot)
			save(join([snapshot, "_", string(j), ".jld"]), "z", reshape(z,J.model.n), "x", reshape(x,J.model.n))
		end
	end

	return x
end


