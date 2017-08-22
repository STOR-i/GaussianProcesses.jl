# Zero mean function

@doc """
# Description
Constructor for the zero mean function

m() = 0
""" ->
type MeanZero <: Mean
end

num_params(mZero::MeanZero) = 0
mean(mZero::MeanZero, x::MatF64) =  fill(0.0, size(x,2))
get_params(mZero::MeanZero) = Float64[]
get_priors(mZero::MeanZero) = []
