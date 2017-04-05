drfun = 'MCDE_KOL_3';

options = [];
options.drfun = drfun;
options.train = [];
options.train.raw = train;
options.train.label = train_label;
options.test = [];
options.test.raw = test;
options.test.label = test_label;

save('options.mat', 'options');

V = length(train);
sigmak_ = [5 6 7 8 9 10 11 12 13 14 15];
dimension_ = [1 2 3 4 5 6 7 8 9 10];
knn = 1;

[dimension, sigmak] = meshgrid(dimension_, sigmak_);
n = numel(sigmak_);

fprintf('ratio 0.1 ... \n');
for i=1:n,
    options.knn = knn;
    options.dimension = 1;
    
    options.alpha = 1;
    options.beta = 1;
    options.sigmak = sigmak_(i);
    R = exp_acmmm(X, L, options);
    fprintf('dimension=%d, sigmak=%d, alpha=%d, beta=%d, f1=%d\n', ...
        dimension(i), sigmak_(i), options.alpha, options.beta, R.f);
    filename = sprintf('W_%d_%d_%d.mat', options.alpha, options.beta, options.sigmak);
    W = R.W;
    save(filename, 'W');
    
    options.alpha = 0.5;
    options.beta = 0.5;
    options.sigmak = sigmak_(i);
    R = exp_acmmm(X, L, options);
    fprintf('dimension=%d, sigmak=%d, alpha=%d, beta=%d, f1=%d\n', ...
        dimension(i), sigmak_(i), options.alpha, options.beta, R.f);
    filename = sprintf('W_%d_%d_%d.mat', options.alpha, options.beta, options.sigmak);
    W = R.W;
    save(filename, 'W');
    
    options.alpha = 0.1;
    options.beta = 0.1;
    options.sigmak = sigmak_(i);
    R = exp_acmmm(X, L, options);
    fprintf('dimension=%d, sigmak=%d, alpha=%d, beta=%d, f1=%d\n', ...
        dimension(i), sigmak_(i), options.alpha, options.beta, R.f);
    filename = sprintf('W_%d_%d_%d.mat', options.alpha, options.beta, options.sigmak);
    W = R.W;
    save(filename, 'W');
end