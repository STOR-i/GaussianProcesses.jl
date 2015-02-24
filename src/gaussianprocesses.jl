module gaussianprocesses
using Distributions

# Functions that should be available to package
# users should be explicitly exported here

export GP, predict, se, exf, gef, mat32, mat52, meanZero, meanConst, EI

# all package code should be included here
include("GP.jl")
include("expected_improvement.jl")
include("mean_functions.jl")
include("kernels.jl")

end # module
