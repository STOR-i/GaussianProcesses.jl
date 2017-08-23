type SumMean <: CompositeMean
    means::Vector{Mean}
    SumMean(args::Vararg{Mean}) = new(collect(args))
end

submeans(sm::SumMean) = sm.means

mean(summean::SumMean, x::MatF64) = sum(mean(m, x) for m in submeans(summean))

get_param_names(summean::SumMean) = composite_param_names(summean.means, :sm)

function grad_mean(summean::SumMean, x::Vector{Float64})
     dm = Array{Float64}(0)
      for m in summean.means
        append!(dm,grad_mean(m, x))
      end
    dm
end

+(m1::SumMean, m2::Mean) = SumMean(m1.means..., m2)
+(m1::SumMean, m2::SumMean) = SumMean(m1.means..., m2.means...)
+(m1::Mean, m2::Mean) = SumMean(m1,m2)
+(m1::Mean, m2::SumMean) = SumMean(m1, m2.means...)
