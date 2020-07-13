module TestCV
    using GaussianProcesses
    using GaussianProcesses: get_params, set_params!, update_mll!, get_value
    using Test, Random
    using Distributions: Normal, Uniform, logpdf, MvNormal
    import Calculus

    @testset "leave-one-out" begin
        Random.seed!(1)
        n,p = 10,1
        f_star(x::Real) = abs(x-5)*cos(2*x)
        σ_y = 0.8
        X_distr = Uniform(-2,2)
        ϵ_distr = Normal(0,σ_y)
        x = sort(rand(X_distr, n))
        Y = f_star.(x) .+ rand(ϵ_distr,n)
        k = SEIso(0.5, 0.8)
        logNoise = log(σ_y)
        m = MeanLin([1.0])
        gp = GPE(Matrix(x'), Y, m, k, logNoise)
        optimize!(gp; domean=false, kern=true, noise=true)

        μi,σi2 = GaussianProcesses.predict_LOO(gp)

        CV = 0.0
        @testset "predictions" begin
            for i in 1:n
                xV, yV = gp.x[:,[i]], gp.y[i]
                T = [j for j in 1:n if j!=i]
                xT, yT = gp.x[:,T], gp.y[T]
                gpT = GPE(xT, yT, gp.mean, gp.kernel, gp.logNoise)
                pred_i = predict_y(gpT, xV)
                @test pred_i[1][1] ≈ μi[i] atol=1e-5
                @test  pred_i[2][1] ≈ σi2[i] atol=1e-5
                CV += logpdf(Normal(μi[i], √σi2[i]), gp.y[i])
            end
        end
        @testset "CVmetric" begin
            @test CV ≈ GaussianProcesses.logp_LOO(gp)
        end
        @testset "gradient" begin
            target = function(θ)
                θprev = get_params(gp.kernel)
                set_params!(gp.kernel, θ)
                update_mll!(gp)
                CV = GaussianProcesses.logp_LOO(gp)
                set_params!(gp.kernel, θprev) # put it back
                return CV
            end
            grad_numerical = Calculus.gradient(target, get_params(k))
            update_mll!(gp)
            grad_analytical = GaussianProcesses.dlogpdθ_LOO(gp; noise=false, kern=true, domean=false)
            @test grad_numerical ≈ grad_analytical  atol=1e-6
        end
        @testset "logNoise gradient" begin
            target = function(θ)
                θprev = get_params(gp; noise=true, kern=false, domean=false)
                set_params!(gp, θ; noise=true, kern=false, domean=false)
                update_mll!(gp)
                CV = GaussianProcesses.logp_LOO(gp)
                set_params!(gp, θprev; noise=true, kern=false, domean=false) # put it back
                return CV
            end
            grad_numerical = Calculus.gradient(target, Float64[get_value(gp.logNoise)])
            update_mll!(gp)
            grad_analytical = GaussianProcesses.dlogpdθ_LOO(gp; noise=true, kern=false, domean=false)
            @test grad_numerical ≈ grad_analytical  atol=1e-6
        end
    end

    @testset "folds" begin
        Random.seed!(1)
        n,p = 20,1
        folds = [1:5, 6:14, 15:20]
        f_star(x::Real) = abs(x-5)*cos(2*x)
        σ_y = 0.8
        X_distr = Uniform(-2,2)
        ϵ_distr = Normal(0,σ_y)
        x = sort(rand(X_distr, n))
        Y = f_star.(x) .+ rand(ϵ_distr,n)
        k = SEIso(0.5, 0.8)
        logNoise = log(σ_y)
        m = MeanLin([1.0])
        gp = GPE(Matrix(x'), Y, m, k, logNoise)
        optimize!(gp; domean=false, kern=true, noise=true)

        μ,Σ = GaussianProcesses.predict_CVfold(gp, folds)

        CV = 0.0
        @testset "predictions" begin
            for (ifold,V) in enumerate(folds)
                xV, yV = gp.x[:,V], gp.y[V]
                T = setdiff(1:n, V)
                xT, yT = gp.x[:,T], gp.y[T]
                gpT = GPE(xT, yT, gp.mean, gp.kernel, gp.logNoise)
                pred_V = predict_y(gpT, xV; full_cov=true)
                @test pred_V[1] ≈ μ[ifold] atol=1e-5
                @test Matrix(pred_V[2]) ≈ Σ[ifold] atol=1e-5
                CV += logpdf(MvNormal(μ[ifold], pred_V[2]), gp.y[V])
            end
        end
        @testset "CVmetric" begin
            @test CV ≈ GaussianProcesses.logp_CVfold(gp, folds) atol=1e-5
        end
        @testset "gradient" begin
            target = function(θ)
                θprev = get_params(gp.kernel)
                set_params!(gp.kernel, θ)
                update_mll!(gp)
                CV = GaussianProcesses.logp_CVfold(gp, folds)
                set_params!(gp.kernel, θprev) # put it back
                return CV
            end
            grad_numerical = Calculus.gradient(target, get_params(k))
            update_mll!(gp)
            grad_analytical = GaussianProcesses.dlogpdθ_CVfold(gp, folds; noise=false, kern=true, domean=false)
            @test grad_numerical ≈ grad_analytical  atol=1e-6
        end
        @testset "logNoise gradient" begin
            target = function(θ)
                θprev = get_params(gp; noise=true, kern=false, domean=false)
                set_params!(gp, θ; noise=true, kern=false, domean=false)
                update_mll!(gp)
                CV = GaussianProcesses.logp_CVfold(gp, folds)
                set_params!(gp, θprev; noise=true, kern=false, domean=false) # put it back
                return CV
            end
            grad_numerical = Calculus.gradient(target, Float64[get_value(gp.logNoise)])
            update_mll!(gp)
            grad_analytical = GaussianProcesses.dlogpdθ_CVfold(gp, folds; noise=true, kern=false, domean=false)
            @test grad_numerical ≈ grad_analytical  atol=1e-6
        end
    end
end
