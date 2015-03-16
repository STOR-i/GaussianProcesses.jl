module GaP
using Distributions

# Functions that should be available to package
# users should be explicitly exported here

export GP, predict, SumKernel, SE, MAT32, MAT52, EXF, PERI, POLY, RQ, LIN, mZERO, mCONST, mLIN, mPOLY, SumMean, EI, optimize!, plotGP, plotEI

# all package code should be included here
include("means/meanFunctions.jl")
include("kernels/kernels.jl")
include("utils.jl")
include("GP.jl")
include("expected_improvement.jl")
include("optimize.jl")
include("plotting.jl")

end # module
