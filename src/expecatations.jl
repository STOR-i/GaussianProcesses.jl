function variational_expectation(ll::PoisLik, y::Vector{Float64}, m::Vector{Float64}, V::Vector{Float64})
    return y*m - exp.(m + V/2) - log(factorial(convert(Int64, y)))
end
