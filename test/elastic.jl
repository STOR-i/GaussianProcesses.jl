module TestElastic
using Test, GaussianProcesses

@testset "elastic" begin
    N = 10; nobs = 20;
    for m in [MeanLin(rand(N)), MeanZero(), MeanConst(rand())]
        for k in [Mat52Iso(rand(), rand()), SEIso(rand(), rand()), 
                  Mat32Ard(rand(N), rand()), RQArd(rand(N), rand(), rand()), 
                  LinIso(rand()), LinArd(rand(N))]
            x0 = Array{Float64, 2}(undef, N, 0); y0 = Array{Float64, 1}(undef, 0);
            x = rand(N, nobs); y = rand(nobs);
            gp1 = GPE(x, y, m, k, -2.)
            gp2 = ElasticGPE(x0, y0, m, k, -2.)
            append!(gp2, x[:, 1:7], y[1:7])
            append!(gp2, x[:, 8], y[8])
            append!(gp2, x[:, 9:20], y[9:20])
            @test gp1.cK.chol.U ≈ view(gp2.cK.chol).U
            xtest = rand(N, 2)
            mu1, sig1 = predict_y(gp1, xtest)
            mu2, sig2 = predict_y(gp2, xtest)
            @test mu1 ≈ mu2
            @test sig1 ≈ sig2
            optimize!(gp1)
            optimize!(gp2)
            @test gp1.mll ≈ gp2.mll
            @test isapprox(gp1.alpha, gp2.alpha, atol = 1e-4)
        end
    end
end

end
