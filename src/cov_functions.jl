# Here will go built-in covariance functions

# Example function - could be improved
exp_dist(x::Vector{Float64}, y::Vector{Float64}) =  exp(-norm(x-y)^2)
