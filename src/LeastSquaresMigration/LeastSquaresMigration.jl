# Sparsity promoting LSRTM w/ the linearized Bregman method
# Module contains main algorithm + preconditioners
# Author: Philipp Witte
# Date: December 2016
#

module LeastSquaresMigration
import Base.norm

using JOLI
using PyPlot
using JLD
using ..TimeModeling

# Files
include("linbreg_lsrtm.jl")
include("model_preconditioners.jl")
include("data_topmute.jl")
include("auxiliary_functions.jl")

end
