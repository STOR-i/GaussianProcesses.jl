#Zero mean function

type mZERO <: Mean
    mZERO() = new()
end    
    
meanf(mZero::mZERO,x::Matrix{Float64}) =  zeros(size(x, 2))
