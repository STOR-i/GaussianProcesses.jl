# Here will go built-in covariance functions

#See Chapter 4 Page 90 of Rasumussen and Williams Gaussian Processes for Machine Learning

# Squared Exponential Function 
se(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}) =  exp(-hyp[2]*norm(x-y)^2/hyp[1]^2)

#Exponential Function
exf(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}) =  exp(-norm(x-y)/hyp^2)

#Gamma Exponential Function
gef(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}) =  exp(-(norm(x-y)/hyp[1])^hyp[2])    

#Matern 3/2 Function
mat32(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}) =  hyp[2]*(1+sqrt(3*norm(x-y)^2)/hyp[1])*exp(-sqrt(3*norm(x-y)^2)/hyp[1])    

#Matern 5/2 Function
mat52(x::Vector{Float64}, y::Vector{Float64}, hyp::Vector{Float64}) =  hyp[2]*(1+sqrt(5*norm(x-y)^2)/hyp[1]+sqrt(5*norm(x-y)^2)/(3*hyp[1]^2))*exp(-sqrt(5*norm(x-y)^2)/hyp[1])    





