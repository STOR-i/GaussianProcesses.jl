#Zero mean function

@doc """
# Description
Constructor for the zero mean function

m(x) = 0
""" ->
type MeanZero <: Mean
    β::Float64
    MeanZero() = new(0.0)
end

mean(mZero::MeanZero,x::Matrix{Float64}) =  fill(0.0, size(x,2))

get_params(mZero::MeanZero) = Float64[]
num_params(mZero::MeanZero) = 0
