using GaussianProcesses
using GaussianProcesses: distance, KernelData
#import ScikitLearnBase

d, n = 3, 10

x = 2π * rand(d, n)
y = Float64[sum(sin(x[:,i])) for i in 1:n]/d
mZero = MeanZero()
kern = SE(0.0,0.0)

gp = GP(x, y, mZero, kern)


# Function verifies that predictive mean at input observations
# are the same as the output observations
function test_pred_matches_obs(gp::GPE)
    y_pred, sig = predict_y(gp, x)
    @test_approx_eq_eps maximum(abs(gp.y - y_pred)) 0.0 1e-4
end

test_pred_matches_obs(gp)

# function sk_test_pred_matches_obs() # ScikitLearn interface test
#     gp_sk = ScikitLearnBase.fit!(GPE(), x', y)
#     y_pred = ScikitLearnBase.predict(gp_sk, x')
#     @test_approx_eq_eps maximum(abs(gp_sk.y - y_pred)) 0.0 1e-4
# end

#sk_test_pred_matches_obs()

# Modify kernel and update
gp.k.ℓ2 = 4.0
x_pred = 2π * rand(d, n)

GaussianProcesses.update_target!(gp)
y_pred, sig = predict_y(gp, x_pred)

#—————————————————————————————————————————–
#GPMC test

# z = Φ(y)

# z = convert(Vector{Bool},z)
# lik = BernLik()

# gp = GP(x, z, mZero, kern, lik)
# z_pred = predict_y(gp, x)[1]
# maximum(abs(gp.y - z_pred)) 0.0 1e-4
