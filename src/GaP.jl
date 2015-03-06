module GaP
using Distributions

# Functions that should be available to package
# users should be explicitly exported here

export GP, predict, SE, MAT32, MAT52, EXF, PERI, POLY, mZERO, EI, optimize!

# all package code should be included here
include("means/meanFunctions.jl")
include("kernels/kernels.jl")
include("utils.jl")
include("GP.jl")
include("expected_improvement.jl")
include("optimize.jl")

end # module
