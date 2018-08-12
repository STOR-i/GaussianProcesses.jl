# Zero mean function

"""
    MeanZero <: Mean

Zero mean function
```math
m(x) = 0.
```
"""
struct MeanZero <: Mean end

num_params(mZero::MeanZero) = 0
grad_mean(mZero::MeanZero, x::VecF64) = Float64[]
Statistics.mean(mZero::MeanZero, x::VecF64) = 0.0
Statistics.mean(mZero::MeanZero, X::MatF64) =  fill(0.0, size(X,2))
get_params(mZero::MeanZero) = Float64[]

function set_params!(mZero::MeanZero, hyp::VecF64)
    length(hyp) == 0 || throw(ArgumentError("Zero mean function has no parameters"))
end

get_priors(mZero::MeanZero) = []
