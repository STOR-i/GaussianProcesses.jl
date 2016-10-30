type FixedKern <: Kernel
    kern::Kernel
    free::Vector{Int} # vector of *free* parameters
end


get_params(fk::FixedKern) = get_params(fk.kern)[fk.free]
get_param_names(fk::FixedKern) = get_param_names(fk.kern)[fk.free]
function set_params!(fk::FixedKern, hyp)
    p = get_params(fk.kern)
    p[fk.free] = hyp
    set_params!(fk.kern, p)
end
num_params(fk::FixedKern) = length(fk.free)

# convenience functions to fix a parameter
function fix(k::Kernel, par::Symbol)
    npars = num_params(k)
    free = collect(1:npars)
    names = get_param_names(k)
    tofix = find(names.==par)[1]
    deleteat!(free, tofix)
    return FixedKern(k, free)
end
function fix(k::FixedKern, par::Symbol)
    free = k.free
    names = get_param_names(k)
    tofix = find(names.==par)[1]
    deleteat!(free, tofix)
    return FixedKern(k.kern, free)
end
function fix(k::Kernel)
    return FixedKern(k, Int[])
end

# convenience functions to free a parameter
function free(k::Kernel)
    return k.kern
end
function free(k::Kernel, par::Symbol)
    all_names = get_param_names(k.kern)
    ipar = find(all_names.==par)[1]
    free = sort(unique([k.free; ipar]))
    if length(free) == num_params(k.kern)
        return k.kern
    else
        return FixedKern(k.kern, free)
    end
end

function grad_slice!{M1<:MatF64,M2<:MatF64}(dK::M1, fk::FixedKern, X::M2, data::KernelData, p::Int)
    return grad_slice!(dK, fk.kern, X, data, fk.free[p])
end
function grad_slice!{M1<:MatF64,M2<:MatF64}(dK::M1, fk::FixedKern, X::M2, data::EmptyData, p::Int)
    return grad_slice!(dK, fk.kern, X, data, fk.free[p])
end

# delegate everything else to the wrapped kernel
# (is there a better way to do this?)
cov(fk::FixedKern, args...) = cov(fk.kern, args...)
cov{M1<:MatF64,M2<:MatF64}(fk::FixedKern, X₁::M1, X₂::M2) = cov(fk.kern, X₁, X₂)
cov{M<:MatF64}(fk::FixedKern, X::M) = cov(fk.kern, X)
cov{M<:MatF64}(fk::FixedKern, X::M, data::EmptyData) = cov(ck.kern,X,data)
KernelData(fk::FixedKern, args...) = KernelData(fk.kern, args...)
KernelData{M<:MatF64}(fk::FixedKern, X::M) = KernelData(fk.kern, X)
kernel_data_key(fk::FixedKern, args...) = kernel_data_key(fk.kern, args...)
cov!(cK::MatF64, fk::FixedKern, args...) = cov!(cK, fk.kern, args...)
cov!{M1<:MatF64,M2<:MatF64}(cK::MatF64, fk::FixedKern, X₁::M1, X₂::M2) = cov!(cK, fk.kern, k, X₁, X₂)
cov!{M<:MatF64}(cK::MatF64, fk::FixedKern, X::M, data::EmptyData)=cov!(cK, fk.kern,X,data)
cov!{M<:MatF64}(cK:: MatF64,fk::FixedKern,X::M) = cov!(cK, fk.kern, X)
addcov!(s, fk::FixedKern, args...) = addcov!(s, fk.kern, args...)
addcov!{M<:MatF64}(cK::MatF64, fk::FixedKern, X::M) = addcov!(cK, fk.kern, X)
multcov!(s, fk::FixedKern, args...) = multcov!(s, fk.kern, args...)
multcov!{M<:MatF64}(cK::MatF64, fk::FixedKern, X::M) = multcov!(cK, fk.kern,X)
function multcov!{M<:AbstractArray{Float64,2}}(
    cK::AbstractArray{Float64,2}, 
    fk::FixedKern, X::M, data::KernelData)
    return multcov!(cK, fk.kern, X, data)
end
function addcov!{M<:AbstractArray{Float64,2}}(
    cK::AbstractArray{Float64,2}, 
    fk::FixedKern, X::M, data::KernelData)
    return addcov!(cK, fk.kern, X, data)
end
