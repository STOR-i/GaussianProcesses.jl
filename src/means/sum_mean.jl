struct SumMean <: CompositeMean
    means::Vector{Mean}
end

SumMean(args::Vararg{Mean}) = SumMean(collect(args))

submeans(sm::SumMean) = sm.means

Statistics.mean(sm::SumMean, x::VecF64) = sum(mean(m, x) for m in submeans(sm))
Statistics.mean(sm::SumMean, X::MatF64) = sum(mean(m, X) for m in submeans(sm))

get_param_names(summean::SumMean) = composite_param_names(summean.means, :sm)

function grad_mean(sm::SumMean, x::VecF64)
    np = num_params(sm)
    dm = Array{Float64}(undef, np)
    v = 1
    for m in sm.means
        np_m = num_params(m)
        dm[v:v+np_m-1] = grad_mean(m, x)
        v+=np_m
    end
    dm
end

Base.:+(m1::SumMean, m2::Mean) = SumMean(m1.means..., m2)
Base.:+(m1::SumMean, m2::SumMean) = SumMean(m1.means..., m2.means...)
Base.:+(m1::Mean, m2::Mean) = SumMean(m1,m2)
Base.:+(m1::Mean, m2::SumMean) = SumMean(m1, m2.means...)
