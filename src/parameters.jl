abstract type Parameter end

mutable struct KernelParameter{A<:Float64, B<:Bool} <: Parameter
    value::A
    trainable::B
end

KernelParameter{A, B}(value, trainable) where {A, B} = KernelParameter(value, trainable)

