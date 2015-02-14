# Here will go built-in covariance functions

# Example function - could be improved
rbf(x::Vector{Float64}, y::Vector{Float64}) =  exp(-0.5 * norm(x-y)^2)
