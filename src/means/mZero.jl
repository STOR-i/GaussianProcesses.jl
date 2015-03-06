#Zero mean function

type mZERO <: Mean
    x::Float64
    mZERO(x::Float64=0.0) = new(x)
end    
    
meanf(mZero::mZERO,x::Matrix{Float64}) =  zeros(size(x, 2))
