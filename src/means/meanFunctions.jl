#This file contains a list of the currently available mean functions

import Base.show

abstract Mean

function show(io::IO, m::Mean, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(m)), Params: ")
    show(io, get_params(m))
    print(io, "\n")
end

include("mConst.jl")         # Constant mean function, which also contains the zero mean function
include("mLin.jl")           # Linear mean function
include("mPoly.jl")          # Polynomial mean function
include("sum_mean.jl")       # Sum mean functions
include("prod_mean.jl")      # Product of mean functions

