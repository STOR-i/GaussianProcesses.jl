module gaussianprocesses

# Functions that should be available to package
# users should be explicitly exported here

export GaussianProcess, predict, rbf

# all package code should be included here
include("GaussianProcesses.jl")
#include("expected_improvement.jl")
#include("mean_functions.jl")
include("kernels.jl")

end # module
