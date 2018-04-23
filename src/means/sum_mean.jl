type SumMean <: CompositeMean
    means::Vector{Mean}
    SumMean(args::Vararg{Mean}) = new(collect(args))
end

submeans(sm::SumMean) = sm.means

mean(sm::SumMean, x::VecF64) = sum(mean(m, x) for m in submeans(sm))
mean(summean::SumMean, X::MatF64) = sum(mean(m, X) for m in submeans(summean))

get_param_names(summean::SumMean) = composite_param_names(summean.means, :sm)

function grad_mean(sm::SumMean, x::VecF64)
    np = num_params(sm)
    dm = Array{Float64}(np)
    v = 1
    for m in sm.means
        np_m = num_params(m)
        dm[v:v+np_m-1] = grad_mean(m, x)
        v+=np_m
    end
    dm
end

+(m1::SumMean, m2::Mean) = SumMean(m1.means..., m2)
+(m1::SumMean, m2::SumMean) = SumMean(m1.means..., m2.means...)
+(m1::Mean, m2::Mean) = SumMean(m1,m2)
+(m1::Mean, m2::SumMean) = SumMean(m1, m2.means...)
