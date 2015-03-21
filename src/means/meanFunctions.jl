#This file contains a list of the currently available mean functions

import Distributions.params

abstract Mean

include("mConst.jl")         #Constant mean function, which also contains the zero mean function
include("mLin.jl")           #Linear mean function
include("mPoly.jl")          #Polynomial mean function
include("sum_mean.jl")       #Sum mean functions
include("prod_mean.jl")      #Product of mean functions
