abstract type MeanFunction end

struct Zero <: MeanFunction end
mean(mZero::Zero, x::AbstractMatrix) =  zeros(Float64, size(x))
