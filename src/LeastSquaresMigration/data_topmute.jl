# Data topmute
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: December 2017
#

export marineTopmute,opMarineTopmute

function marineTopmute(Dobs::joData, muteStart::Integer; mute=Array{Any}(3))
	# Data topmute for end-on spread marine streamer data
	Din = deepcopy(Dobs)

	# Design mute window
	j=1
	x0 = 1f0
	xend = length(Din.geometry.xloc[j])
	nt = Din.geometry.nt[j]
	nrec = length(Din.geometry.xloc[j])
	drec = abs(Din.geometry.xloc[j][1] - Din.geometry.xloc[j][2])
	offsetDirectWave = 1.5f0*Din.geometry.t[j]
	idxOffset = Int(round(offsetDirectWave/drec))
	dx = round(idxOffset - idxOffset/10f0)

	if j==1 && ~isassigned(mute)
		z0 = muteStart - Int(round(muteStart/10))
		slope = 0.95f0*(nt - z0)/dx
		mute[1] = x0
		mute[2] = z0
		mute[3] = slope
	elseif j==1 && isassigned(mute)
		x0 = mute[1]
		z0 = mute[2]
		slope = mute[3]
	end
	
	mask = ones(Float32,nt,nrec)
	mask[1:z0,:]=0f0

	# Linear mute
	if (nrec-x0 < dx)
		x = nrec
		zIntercept = Int(round(z0+slope*(x-x0)))
		zax = z0+1:1:zIntercept
	else
		x = x0+dx
		zax = z0+1:1:nt
	end
	if length(zax) > 1
		xax = Array{Int}(round.(linspace(x0,x,length(zax))))
	else 
		xax = Int(round(x0))
	end						    
	for k=1:length(zax)
		mask[zax[k],xax[k]:end] = 0f0
	end

	for j=1:Din.nsrc
		Din.data[j] = Din.data[j].*mask
	end
	return Din
end

function opMarineTopmute(muteStart,geometry;params=Array{Any}(3))
# JOLI wrapper for the linear depth scaling function
	nsrc = length(geometry.xloc)
	N = 0
	for j=1:nsrc
		N += geometry.nt[j]*length(geometry.xloc[j])
	end
	D = joLinearFunctionFwdT(N,N,
							 v -> marineTopmute(v,muteStart;mute=params),
 							 w -> marineTopmute(w,muteStart;mute=params),
							 Float32,Float32,name="Data mute")
	return D
end




