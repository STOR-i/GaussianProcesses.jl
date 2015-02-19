# Here will go the built-in mean functions

#Zero mean function 
meanZero(x::Matrix{Float64}) =  zeros(size(x,2))

#Constant mean function
meanConst(x::Matrix{Float64},b::Float64=1.0) =  b*ones(size(x,2))


