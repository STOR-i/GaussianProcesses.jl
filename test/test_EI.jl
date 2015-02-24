using gaussianprocesses

d, n = 10, 20
x = 2π * rand(d, n)
y = Float64[sum(sin(x[:,i])) for i in 1:n]/d
gp = GP(x, y, meanZero, se, [1.0,0.5])

# Function verifies that predictive mean at input observations
# are the same as the output observations
function test_EI_zero_at_obs(gp::GP)
    ei = EI(gp, gp.x)
    @test maximum(ei) < 1e-4
end

test_pred_matches_obs(gp)
