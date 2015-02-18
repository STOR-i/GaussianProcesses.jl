# Here will go built-in covariance functions

# Radial Basic Function 
rbf(x::Vector{Float64}, y::Vector{Float64}, l::Float64=1.0) =  exp(-0.5 * norm(x-y)^2/l^2)

#Exponential Function
exf(x::Vector{Float64}, y::Vector{Float64}, l::Float64=1.0) =  exp(-norm(x-y)/l^2)

#Gamma Exponential Function
gef(x::Vector{Float64}, y::Vector{Float64}, l::Float64=1.0, gamma::Float64=1.0) =  exp(-(norm(x-y)/l)^gamma)    

#Matern 3/2 Function
mat32(x::Vector{Float64}, y::Vector{Float64}, l::Float64=1.0) =  (1+sqrt(3)*norm(x-y)/l)*exp(-sqrt(3)*norm(x-y)/l)    

#Matern 5/2 Function
mat52(x::Vector{Float64}, y::Vector{Float64}, l::Float64=1.0) =  (1+sqrt(5)*norm(x-y)/l+sqrt(5)*norm(x-y)^2/(3*l^2))*exp(-sqrt(5)*norm(x-y)/l)    





