using GaussianProcesses
using GaussianProcesses: get_params, get_param_names

d, n = 2, 20
ll = rand(d)
x = 2π * rand(d, n)
y = randn(n) + 0.5

""" Not much of a test really... just checks that it doesn't crash
    and that the final mll is better that the initial value
"""
function test_optim(kern::Kernel, X::Matrix{Float64}, y::Vector{Float64})
	gp = GP(X,y,MeanConst(0.0), kern, -3.0)
	init_mLL = gp.mLL
	optimize!(gp)
	@assert gp.mLL > init_mLL
end

function test_fixed(kern::Kernel, X::Matrix{Float64}, y::Vector{Float64})
    init_param = get_params(kern)[1]
    fixed=fix(kern, get_param_names(kern)[1])
	gp = GP(X,y,MeanConst(0.0), fixed, -3.0)
	init_mLL = gp.mLL
	optimize!(gp)
	@assert gp.mLL > init_mLL
	@assert get_params(kern)[1] == init_param
end

se = SEIso(1.0, 1.0)
test_optim(se, x, y)
rq = RQIso(1.0, 1.0, 1.0)
test_fixed(se, x, y)
