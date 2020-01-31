struct ClfMetric{T<:Real}
    N::T
    # Statistics for true data
    Npos::T
    Nneg::T
    # Statistics for predicticted values
    Ppos::T
    Pneg::T
    # True/False Positives
    Tpos::T
    Fpos::T
    # True/False Negatives
    Tneg::T
    Fneg::T
end

function classification(preds::AbstractArray, truth::AbstractArray, threshold::Real=0.5)
    Ppos = 0.0
    Pneg = 0.0
    Tpos = 0.0
    Fpos = 0.0
    Tneg = 0.0
    Fneg = 0.0

    for (pr, tr) in zip(preds, truth)
        if pr > threshold
            Ppos += 1
            if tr == 1
                Tpos += 1
            elseif tr == 0
                Fpos += 1
            else
                throw(BoundsError("True values must be either 0 or 1."))
            end
        elseif pr <= threshold
            Pneg += 1
            if tr == 1
                Fneg += 1
            elseif tr == 0
                Tneg += 1
            else
                throw(BoundsError("True values must be either 0 or 1."))
            end
        end
    end
    N = Float64(length(preds))
    Npos = sum(preds)
    Nneg = N-Npos
    return ClfMetric(N, Npos, Nneg, Ppos, Pneg, Tpos, Fpos, Tneg, Fneg)
end

recall(metric::ClfMetric) = metric.Tpos/(metric.Tpos+metric.Fneg)
accuracy(metric::ClfMetric) = (metric.Tpos + metric.Tneg) / metric.N
precision(metric::ClfMetric) = metric.Tpos/(metric.Tpos + metric.Fpos)

accuracy(pred::AbstractArray, truth::AbstractArray) = 100*sum(pred .== reshape(truth, size(pred)))/size(pred, 2)
function accuracy(gp::GPA, latents::AbstractArray, xTest::AbstractArray, yTest::AbstractArray)
    preds = evaluate(gp, latents, xTest, yTest)
    return accuracy(preds, yTest)
end

function mode(a)
    isempty(a) && throw(ArgumentError("mode is not defined for empty collections"))
    cnts = Dict{eltype(a),Int}()
    # first element
    mc = 1
    mv, st = iterate(a)
    cnts[mv] = 1
    # find the mode along with table construction
    y = iterate(a, st)
    while y !== nothing
        x, st = y
        if haskey(cnts, x)
            c = (cnts[x] += 1)
            if c > mc
                mc = c
                mv = x
            end
        else
            cnts[x] = 1
            # in this case: c = 1, and thus c > mc won't happen
        end
        y = iterate(a, st)
    end
    return mv
end

function proba(gp::GPA, latents::AbstractArray, xTest::AbstractArray)
    probs = Array{Float64}(undef,size(latents, 2), size(xTest,1));
    for i in 1:size(latents, 2)
        set_params!(gp, latents[:,i])         # Set the GP parameters to the posterior values
        update_target!(gp)                   # Update the GP function with the new parameters
        probs[i,:] = predict_y(gp, xTest')[1] # Store the predictive mean
    end
    return probs
end

function evaluate(gp::GPA, latents::AbstractArray, xTest::AbstractArray)
    #Sample from the predictive posterior
    probs = proba(gp, latents, xTest)
    preds = Array{Float64}(undef, 1 ,size(probs, 2));
    for i in 1:size(probs, 2)
        preds[1, i] = mode(round.(probs[:, i]))
    end
    return preds
end

function evaluate(probs::AbstractArray, yTest::AbstractArray)
    preds = Array{Float64}(undef, 1 ,size(probs, 2));
    for i in 1:size(probs, 2)
        preds[1, i] = mode(round.(probs[:, i]))
    end
    return preds
end

rmse(pred, truth) = sqrt(mean((truth.-pred').^2))

mutable struct Chain{T<:AbstractArray, V<:AbstractArray}
    samples::T
    grads::T
    hist_samples::V
    hist_grads::V
end

function chain(samples::AbstractArray, grads::AbstractArray)
    s = [i for i in size(samples)]
    append!(s, 1)
    s = tuple(s...)
    hist_samples = reshape(samples, s)
    hist_grads = reshape(grads, s)
    return Chain(samples, grads, hist_samples, hist_grads)
end

function chain(samples::AbstractArray)
    grads = zeros(size(samples))
    return chain(samples, grads)
end

function push!(a::Chain, b::AbstractArray; grads::Bool=false)
    last_dim = ndims(a.hist_samples)
    if grads
        a.grads = b
        a.hist_grads = cat(a.hist_grads, b, dims=last_dim)
    else
        a.samples = b
        a.hist_samples = cat(a.hist_samples, b, dims=last_dim)
    end
end

function _plot_history(vals::AbstractArray, loop_axis::Int; title::String="")
    ndim = ndims(vals)
    idxs = collect(1:ndim)
    filter!(e->e ≠ loop_axis, idxs)
    append!(idxs, loop_axis)
    vals = permutedims(vals, tuple(idxs...))
    val_set = selectdim(vals, ndim, 1)
    p = plot(val_set', linecolor=[:blue :green :red], xlabel="Iteration", ylabel="Parameter value", title=title, alpha=0.5, leg=false)
    for i in 2:size(vals, ndim)
        val_set = selectdim(vals, ndim, i)
        p = plot!(val_set', linecolor=[:blue :green :red], alpha=0.5, leg=false)
    end
    return p
end

function plot(a::Chain)
    p1 = _plot_history(a.hist_grads, 2; title="Gradient history")
    p2 = _plot_history(a.hist_samples, 2; title="Parameter value history")
    plot(p1, p2)
end
