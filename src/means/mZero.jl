# Zero mean function

@doc """
# Description
Constructor for the zero mean function

m() = 0
""" ->
type MeanZero <: Mean
end

num_params(mZero::MeanZero) = 0
grad_mean(mZero::MeanZero, x::Vector{Float64}) = Float64[]
mean(mZero::MeanZero, x::VecF64) = 0.0
mean(mZero::MeanZero, X::MatF64) =  fill(0.0, size(X,2))
get_params(mZero::MeanZero) = Float64[]

function set_params!(mZero::MeanZero, hyp::Vector{Float64})
    length(hyp) == 0 || throw(ArgumentError("Zero mean function has no parameters"))
end

get_priors(mZero::MeanZero) = []
