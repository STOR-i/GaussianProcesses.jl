struct SumMean{T<:NTuple{N,Mean} where N} <: CompositeMean
    means::T
end

SumMean(means::Mean...) = SumMean(means)

mean(sm::SumMean, x::AbstractVector) = sum(mean(m, x) for m in components(sm))

get_param_names(sm::SumMean) = composite_param_names(components(sm), :sm)

function grad_mean(sm::SumMean, x::AbstractVector)
    dm = Array{Float64}(undef, num_params(sm))
    v = 1
    for m in components(sm)
        w = v + num_params(m)
        dm[v:(w - 1)] = grad_mean(m, x)
        v = w
    end
    dm
end

Base.:+(m1::SumMean, m2::Mean) = SumMean(m1.means..., m2)
Base.:+(m1::SumMean, m2::SumMean) = SumMean(m1.means..., m2.means...)
Base.:+(m1::Mean, m2::Mean) = SumMean(m1, m2)
Base.:+(m1::Mean, m2::SumMean) = SumMean(m1, m2.means...)
