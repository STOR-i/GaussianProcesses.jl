import Base.append!
append!(gp, x::VecF64, y::Float64) = append!(gp, reshape(x, :, 1), [y])
function append!(gp::GPE{X,Y,M,K,P,D}, x::MatF64, y::VecF64) where {X,Y,M,K,P <: ElasticPDMat, D}
    size(x, 2) == length(y) || error("$(size(x, 2)) observations, but $(length(y)) targets.")
    newcov = [cov(gp.kernel, gp.x, x); cov(gp.kernel, x, x) + (exp(2*gp.logNoise) + 1e-5)*I]
    append!(gp.x, x)
    append!(gp.cK, newcov)
    gp.nobs += length(y)
    append!(gp.y, y)
    update_mll!(gp, kern = false, noise = false)
end

init_cK(::Type{<:ElasticPDMat}, m) = ElasticPDMat(m)
wrap_cK(cK::ElasticPDMat, Σbuffer, chol) = cK
mat(cK::ElasticPDMat) = view(cK.mat)
cholfactors(cK::ElasticPDMat) = view(old_cK.chol).factors

is_kerneldata_updated(gp::GPE{X,Y,M,K,P,D}) where {X,Y,M,K,P <: ElasticPDMat,D} = false

