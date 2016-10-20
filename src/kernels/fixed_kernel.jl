type FixedKern <: Kernel
    kern::Kernel
    free::Vector{Int} # vector of *free* parameters
end

cov(fk::FixedKern, args...) = cov(fk.kern, args...)
KernelData(fk::FixedKern, args...) = KernelData(fk.kern, args...)
kernel_data_key(fk::FixedKern, args...) = kernel_data_key(fk.kern, args...)
cov!(fk::FixedKern, args...) = cov!(fk.kern, args...)
addcov!(s, fk::FixedKern, args...) = addcov!(s, fk.kern, args...)
multcov!(s, fk::FixedKern, args...) = multcov!(s, fk.kern, args...)
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
grad_slice!{M1<:MatF64}(dK::M1, fk::FixedKern, args...) = grad_slice!(dK, fk.kern, args...)

get_params(fk::FixedKern) = get_params(fk.kern)[fk.free]
get_param_names(fk::FixedKern) = get_param_names(fk.kern)[fk.free]
function set_params!(fk::FixedKern, hyp)
    p = get_params(fk.kern)
    p[fk.free] = hyp
    set_params!(fk.kern, p)
end
num_params(fk::FixedKern) = length(fk.free)

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
