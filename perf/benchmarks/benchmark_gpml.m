run('gpml-matlab-v4.1-2017-10-19/startup.m')

rng(1);


kernels = containers.Map;
kernels('se') = @covSEiso;
kernels('mat12') = {@covMaterniso, 1};
kernels('rq') = @covRQiso;
kernels('se+rq') = { 'covSum', { 'covSEiso', 'covRQiso' } };
kernels('se*rq') = { 'covProd', { 'covSEiso', 'covRQiso' } };
kernels('se+se2+rq') = { 'covSum', { 'covSEiso', 'covSEiso', 'covRQiso' } };
kernels('(se+se2)*rq') = {
    'covProd', {
        { 'covSum', { 'covSEiso', 'covRQiso' } }, 
        'covSEiso'}
    };
kernels('mask(se, [1])') = {@covMask, {1,@covSEiso}};
kernels('mask(se, [1])+mask(rq, [2:10])') = { 
    @covSum, { 
        {@covMask, {1, @covSEiso}}, 
        {@covMask, {2:dim, @covRQiso}} 
        } 
    };
% kernels('fix(se, σ)') = ?

params = containers.Map;
params('se') = [0 0];
params('mat12') = [0 0]; % ?
params('rq') = [0 0 0];
params('se+rq') = [0 0 0 0 0];
params('se*rq') = [0 0 0 0 0];
params('se+se2+rq') = [0 0 0 0 0 0 0];
params('(se+se2)*rq') = [0 0 0 0 0 0 0];
params('mask(se, [1])') = [0 0];
params('mask(se, [1])+mask(rq, [2:10])') = [0 0 0 0 0];

kernel_names = keys(kernels);
num_kern = length(kernel_names);
timings = zeros(num_kern, 1);

for ikern=1:num_kern
    kname = kernel_names{ikern};
    nexpt = 10; % best of 10
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
    nobsv=3000;
    dim=10;
    hyp = struct('mean', [], 'cov', par, 'lik', 0);
    X = randn(nobsv, dim);
    Y = randn(nobsv, 1);
    tic();
    gp(hyp, @infGaussLik, meanfunc, kern, likfunc, X, Y);
    t = toc();
end
