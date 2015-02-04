module gaussianprocesses

# Functions that should be available to package
# users should be explicitly exported here
export exp_dist

# all package code should be included here
include("gauss.jl")
include("expected_improvement.jl")
include("mean_functions.jl")
include("cov_functions.jl")

end # module
