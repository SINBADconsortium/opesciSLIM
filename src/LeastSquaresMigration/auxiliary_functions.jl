# Functions for LSRTM
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: December 2016
#

export softThresholding, subsample, Snapshot

mutable struct Snapshot
	path::String
	filename::Array{Any,1}
end

function Snapshot(path::String, filenames...)
	filename_cell = Array{Any}(length(filenames))
	for j=1:length(filenames)
		filename_cell[j] = filenames[j]
	end
	return Snapshot(path, filename_cell)
end

function softThresholding(x, lambda)
# Pointwise soft thresholding function
	return xThresh = sign.(x).*max.(abs.(x)-lambda, 0.f0)
end


