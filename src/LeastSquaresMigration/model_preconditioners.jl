# Model-space preconditioners for LSRTM
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: December 2016
#

export model_topmute, opTopmute, findWaterBottom, depth_scaling, opDepthScaling, laplace

function model_topmute(n::Tuple{Int64,Int64},mute_end::Array{Integer,1},length::Int64,x)
# Model domain topmute for a velocity model of dimensions n = [nx, nz].
	x = reshape(x,n)
	for j=1:n[1]
		mute_start = mute_end[j] - length
		filter = zeros(Float32,n[2])
		filter[1:mute_start-1] = 0f0
		filter[mute_end[j]+1:end] = 1f0
		taper_length = mute_end[j] - mute_start + 1
		taper = (1f0 + sin.((pi*(0:taper_length-1))/(taper_length - 1)-pi/2f0))/2f0
		filter[mute_start:mute_end[j]] = taper
		M = spdiagm(filter)
		x[j,:] = x[j,:].*filter
	end
	return vec(x)
end


function model_topmute(n::Tuple{Int64,Int64},mute_end::Int64,length::Int64,x)
# Model domain topmute for a velocity model of dimensions n = [nx, nz].
	x = reshape(x,n)
	mute_start = mute_end - length
	filter = zeros(Float32,n[2])
	filter[1:mute_start-1] = 0f0
	filter[mute_end+1:end] = 1f0
	taper_length = mute_end - mute_start + 1
	taper = (1f0 + sin.((pi*(0:taper_length-1))/(taper_length - 1)-pi/2f0))/2f0
	filter[mute_start:mute_end] = taper
	M = spdiagm(filter)
	for j=1:n[1]
		x[j,:] = x[j,:].*filter
	end
	return vec(x)
end


function opTopmute(n,mute_start,length)
	# JOLI wrapper for model domain topmute
	N = prod(n)
	T = joLinearFunctionFwdT(N,N,
 							 v -> model_topmute(n,mute_start,length,v),
 							 w -> model_topmute(n,mute_start,length,w),
							 Float32,Float32,name="Model topmute")
	return T
end


function findWaterBottom(m)
	#return the indices of the water bottom of a seismic image
	n = size(m)
	idx = zeros(Integer,n[1])
	eps = 1e-4
	for j=1:n[1]
		k=1
		while true
			if abs(m[j,k]) > eps
				idx[j] = k
				break
			end
			k += 1
		end
	end
	return idx
end


function depth_scaling(m,model)
# Linear depth scaling function for seismic images
	m = reshape(m,model.n)
	filter = sqrt(0f0:model.d[2]:model.d[2]*(model.n[2]-1))
	F = spdiagm(filter)
	for j=1:model.n[1]
		m[j,:] = F*m[j,:]
	end
	return vec(m)
end

function opDepthScaling(model)
# JOLI wrapper for the linear depth scaling function
	N = prod(model.n)
	D = joLinearFunctionFwdT(N, N,
							 v -> depth_scaling(v,model),
 							 w -> depth_scaling(w,model),
							 Float32,Float32,name="Depth scaling")
end


function laplace(model::TimeModeling.Model)
# 2D Laplace operator

	# 2nd derivative in x direction
	d1 = ones(Float32,model.n[1]-1)*1f0/(model.d[1]^2)
	d2 = ones(Float32,model.n[1])*-2f0/(model.d[1]^2)
	d3 = d1
	d = (d1, d2, d3)
	position = (-1, 0, 1)

	Dx = spdiagm(d,position,model.n[1],model.n[1])
	Ix = speye(Float32,model.n[1])

	# 2nd derivative in z direction
	d1 = ones(Float32,model.n[2]-1)*1f0/(model.d[2]^2)
	d2 = ones(Float32,model.n[2])*-2f0/(model.d[2]^2)
	d3 = d1
	d = (d1, d2, d3)
	position = (-1, 0, 1)

	Dz = spdiagm(d,position,model.n[2],model.n[2])
	Iz = speye(Float32,model.n[2])

	# 2D Laplace operator
	D = kron(Dz,Ix) + kron(Iz,Dx)
end





