import Distributions.params

# Here will go built-in covariance functions

#See Chapter 4 Page 90 of Rasumussen and Williams Gaussian Processes for Machine Learning

abstract Kernel

type SE <: Kernel
    l::Float64
    σ::Float64
    SE(l::Float64=1.0, σ::Float64=0.5) = new(l, σ)
end


kern(se::SE, x::Vector{Float64}, y::Vector{Float64}) = exp(-se.σ*norm(x-y)^2/se.l^2)
params(se::SE) = (se.l, se.σ)


#Exponential Function
exf(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}) =  exp(-norm(x-y)/hyp^2)

#Gamma Exponential Function
gef(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}) =  exp(-(norm(x-y)/hyp[1])^hyp[2])    

#Matern 3/2 Function
mat32(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}=[1.0,1.0]) =  hyp[2]*(1+sqrt(3*norm(x-y)^2)/hyp[1])*exp(-sqrt(3*norm(x-y)^2)/hyp[1])    

#Matern 5/2 Function
mat52(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}=[1.0,1.0]) =  hyp[2]*(1+sqrt(5*norm(x-y)^2)/hyp[1]+sqrt(5*norm(x-y)^2)/(3*hyp[1]^2))*exp(-sqrt(5*norm(x-y)^2)/hyp[1])    





