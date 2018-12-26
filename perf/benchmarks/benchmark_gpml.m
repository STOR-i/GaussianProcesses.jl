% run('gpml-matlab-v4.1-2017-10-19/startup.m')
run('gpml-matlab-v4.2-2018-06-11/startup.m')

rng(1);


kernels = containers.Map;
kernels('se') = @covSEiso;
kernels('mat12') = {@covMaterniso, 1};
kernels('rq') = @covRQiso;
kernels('se+rq') = { 'covSum', { 'covSEiso', 'covRQiso' } };
kernels('se*rq') = { 'covProd', { 'covSEiso', 'covRQiso' } };
kernels('se+mat12') = { 'covSum', { 'covSEiso', {@covMaterniso, 1} } };
kernels('se*mat12') = { 'covProd', { 'covSEiso', {@covMaterniso, 1} } };
kernels('se+mat12+rq') = { 'covSum', { 'covSEiso', {@covMaterniso, 1}, 'covRQiso' } };
kernels('(se+mat12)*rq') = {
    'covProd', {
        { 'covSum', { 'covSEiso', {@covMaterniso, 1} } }, 
        'covRQiso'}
    };
kernels('mask(se, [1])') = {@covMask, {1,@covSEiso}};
kernels('mask(se, [1])+mask(rq, [2:10])') = { 
    @covSum, { 
        {@covMask, {1, @covSEiso}}, 
        {@covMask, {2:10, @covRQiso}}  % mask out the first dimension
        } 
    };
% kernels('fix(se, σ)') = ?

params = containers.Map;
params('se') =            [0.3 0.3];
params('mat12') =         [0.3 0.3]; % ?
params('rq') =            [0.3 0.3 0.3];
params('se+rq') =         [0.3 0.3 0.3 0.3 0.3];
params('se*rq') =         [0.3 0.3 0.3 0.3 0.3];
params('se+mat12') =      [0.3 0.3 0.3 0.3];
params('se*mat12') =      [0.3 0.3 0.3 0.3];
params('se+mat12+rq') =   [0.3 0.3 0.3 0.3 0.3 0.3 0.3];
params('(se+mat12)*rq') = [0.3 0.3 0.3 0.3 0.3 0.3 0.3];
params('mask(se, [1])') = [0.3 0.3];
params('mask(se, [1])+mask(rq, [2:10])') = [0.3 0.3 0.3 0.3 0.3];

kernel_names = keys(kernels);
num_kern = length(kernel_names);
timings = zeros(num_kern, 1);


for ikern=1:num_kern
    kname = kernel_names{ikern};
    nexpt = 20; % best of 20
    t = time_GP_bestofn(kernels(kname), params(kname), nexpt);
    timings(ikern) = t;
end

T = table(timings*1000, 'rowNames', kernel_names)
writetable(T, 'bench_results/gpml.csv', 'WriteRowNames', true, 'WriteVariableNames', false) % no column headers

function bestt = time_GP_bestofn(kern, par, nexpt)
    times = zeros(nexpt, 1);
    for i=1:nexpt
        t_i = time_GP(kern, par);
        times(i) = t_i;
    end
    times = sort(times);
    bestt = times(1);
end
function t = time_GP(kern, par)
    meanfunc = [];                    % empty: don't use a mean function
    likfunc = @likGauss;              % Gaussian likelihood
    hyp = struct('mean', [], 'cov', par, 'lik', 0.3);

    dim=10;
    XY_data = csvread("simdata.csv", 1);
    X = XY_data(:, 1:dim);
    Y = XY_data(:, dim+1);
    nobsv = length(Y);

    tic();
    [nlZ dnlZ] = gp(hyp, @infGaussLik, meanfunc, kern, likfunc, X, Y);
    t = toc();
end
