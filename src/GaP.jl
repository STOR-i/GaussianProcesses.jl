module GaP
using Distributions

VERSION < v"0.4-" && using Docile

@document

# Functions that should be available to package
# users should be explicitly exported here

export GP, predict, SumKernel, ProdKernel, SE, MAT32, MAT52, EXF, PERI, POLY, RQ, LIN, mZERO, mCONST, mLIN, mPOLY, SumMean, ProdMean, EI, optimize!, plotGP, plotEI

# all package code should be included here
include("means/meanFunctions.jl")
include("kernels/kernels.jl")
include("utils.jl")
include("GP.jl")
include("expected_improvement.jl")
include("optimize.jl")

# This approach to loading supported plotting packages is taken directly from the "KernelDensity" package
macro glue(pkg)
    path = joinpath(dirname(Base.source_path(nothing)),"glue",string(pkg,".jl"))
    init = symbol(string(pkg,"_init"))
    quote
        $(esc(init))() = include($path)
        isdefined(Main,$(QuoteNode(pkg))) && $(esc(init))()
    end
end

@glue Gadfly

end # module
