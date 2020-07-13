struct ProdMean{T<:NTuple{N,Mean} where N} <: CompositeMean
    means::T
end

ProdMean(means::Mean...) = ProdMean(means)

mean(pm::ProdMean, x::AbstractVector) = prod(mean(m, x) for m in components(pm))

get_param_names(pm::ProdMean) = composite_param_names(components(pm), :pm)

function grad_mean(pm::ProdMean, x::AbstractVector)
    dm = Array{Float64}(undef, num_params(pm))
    means = [mean(m, x) for m in components(pm)]
    v = 1
    for (i, m) in enumerate(components(pm))
        p = prod(m2 for (j, m2) in enumerate(means) if i != j)
        w = v + num_params(m)
        dm[v:(w - 1)] = p * grad_mean(m, x)
        v = w
    end
    dm
end

# Multiplication operators
Base.:*(m1::ProdMean, m2::Mean) = ProdMean(m1.means..., m2)
Base.:*(m1::ProdMean, m2::ProdMean) = ProdMean(m1.means..., m2.means...)
Base.:*(m1::Mean, m2::Mean) = ProdMean(m1, m2)
Base.:*(m1::Mean, m2::ProdMean) = ProdMean(m1, m2.means...)
