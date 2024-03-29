# Module with optimization functions for Julia TimeModelling
# Author: Mathias Louboutin, mloubout@eos.ubc.ca
# Date: January, 2017
#

module SLIM_optim
using PolynomialRoots

#############################################################################
# Optimization algorithms
include("SPGSlim.jl")	# minConf_SPG
include("PQNSlim.jl")	# minConf_PQN
include("OptimizationFunctions.jl") # common functions
include("Constraints.jl") # Constriaints projection
end



