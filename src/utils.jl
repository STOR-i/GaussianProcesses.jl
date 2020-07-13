# Transformations
abstract type Transform end

struct LogTransform<:Transform
    forward::Function
    backward::Function
end
LogTransform()=LogTransform(log, exp)
