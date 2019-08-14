module opesciSLIM

using PyCall

# function to prepend Python paths
function prependMyPyPath(d::String)
    mypath=Pkg.dir("opesciSLIM")
    myd=joinpath(mypath,d)
    unshift!(PyVector(pyimport("sys")["path"]), myd)
end

# prepend Python paths
prependMyPyPath("Python/operators/acoustic")
prependMyPyPath("Python/containers")

# submodule TimeModeling
include("TimeModeling/TimeModeling.jl")

# submodule LeastSquaresMigration
include("LeastSquaresMigration/LeastSquaresMigration.jl")

# submodule Optimization
include("Optimization/SLIM_optim.jl")

end
