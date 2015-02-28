import Distributions.params

# Here will go built-in covariance functions

#See Chapter 4 Page 90 of Rasumussen and Williams Gaussian Processes for Machine Learning

abstract Kernel

#Squared Exponential Function
type SE <: Kernel
    l::Float64      # Length scale 
    σ²::Float64     # Signal variance
    SE(l::Float64=1.0, σ²::Float64=0.5) = new(l, σ²)
end

kern(se::SE, x::Vector{Float64}, y::Vector{Float64}) = se.σ²*exp(-0.5*norm(x-y)^2/se.l^2)
params(se::SE) = (se.l, se.σ²)
grad_kern(se::SE, x::Vector{Float64}, y::Vector{Float64}) = [se.σ²*norm(x-y)^2/se.l^3*exp(-0.5*norm(x-y)^2/se.l^2), 2*se.σ*exp(-0.5*norm(x-y)^2/se.l^2)]

#Matern 3/2 Function
type MAT32 <: Kernel
    l::Float64      #Length scale 
    σ²::Float64     #Signal variance
    MAT32(l::Float64=1.0, σ²::Float64=1.0) = new(l, σ²)
end

kern(mat32::MAT32, x::Vector{Float64}, y::Vector{Float64}) = mat32.σ²*(1+sqrt(3*norm(x-y)^2)/mat32.l)*exp(-sqrt(3*norm(x-y)^2)/mat32.l)
params(mat32::MAT32) = (mat32.l, mat32.σ²)


#Matern 5/2 Function
type MAT52 <: Kernel
    l::Float64      #Length scale 
    σ²::Float64     #Signal variance
    MAT52(l::Float64=1.0, σ²::Float64=1.0) = new(l, σ²)
end

kern(mat52::MAT52, x::Vector{Float64}, y::Vector{Float64}) = mat52.σ²*(1+sqrt(5*norm(x-y)^2)/mat52.l+sqrt(5*norm(x-y)^2)/(3*mat52.l^2))*exp(-sqrt(5*norm(x-y)^2)/mat52.l)   
params(mat52::MAT52) = (mat52.l, mat52.σ²)






