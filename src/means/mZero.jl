# Zero mean function

"""
    MeanZero <: Mean

Zero mean function
```math
m(x) = 0.
```
"""
struct MeanZero <: Mean end

num_params(::MeanZero) = 0
grad_mean(::MeanZero, ::AbstractVector) = Float64[]
mean(::MeanZero, ::AbstractVector) = 0.0
mean(mZero::MeanZero, X::AbstractMatrix) =  fill(0.0, size(X, 2))
get_params(::MeanZero) = Float64[]
get_param_names(::MeanZero) = Symbol[]

function set_params!(::MeanZero, hyp::AbstractVector)
    length(hyp) == 0 || throw(ArgumentError("Zero mean function has no parameters"))
end
