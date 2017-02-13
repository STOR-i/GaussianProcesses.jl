"""
# Description
Constructor for the Constant kernel 

k(x,x') = σ²
# Arguments:
* `lσ::Float64`: Log of σ
"""
type Const <: Kernel
    σ2::Float64      # Signal std
    Const(lσ::Float64) = new(exp(2*lσ))
end

function set_params!(cons::Const, hyp::Vector{Float64})
    length(hyp) == 1 || throw(ArgumentError("Constant kernel only has one parameters"))
    cons.σ2 = exp(2.0*hyp)
end

get_params(cons::Const) = Float64[log(cons.σ2)/2.0]
get_param_names(cons::Const) = [:lσ]
num_params(cons::Const) = 1

cov(cons::Const,dim::Vector{Int64}) = cons.σ2*ones(dim)
function cov{V1<:VecF64,V2<:VecF64}(cons::Const, x::V1, y::V2)
    return cov(cons, [length(x),length(y)])
end


@inline dk_dlσ(cons::Const, dim::Vector{Int64}) = 2.0*cov(cons,dim)
@inline function dKij_dθp{M<:MatF64}(cons::Const, X::M, i::Int, j::Int, p::Int, dim::Int)
    return dk_dlσ(cons, [length(X[:,i]),length(X[:,j])])
end
@inline function dKij_dθp{M<:MatF64}(cons::Const, X::M, data::EmptyData, i::Int, j::Int, p::Int, dim::Int)
    return dKij_dθp(cons, X, i, j, p, dim)
end
