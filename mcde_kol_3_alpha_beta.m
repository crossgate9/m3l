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

alpha = [1 0.5 0.1];
beta = [1 0.5 0.1];

fprintf('ratio 0.1 ... \n');
for i=1:n,
    options.fscore_beta = 2;
    options.knn = knn;
    options.dimension = 1;
    options.sigmak = sigmak_(i);

    for j=1:3,
        options.alpha = alpha(j);
        options.beta = beta(j);
        filename = sprintf('%s/W_%d_%d_%d.mat', drfun, options.alpha, options.beta, options.sigmak);
        if exist(filename),
            use_cache = true;
        else
            use_cache = false;
        end

        if use_cache,
            load(filename);
            options.W = W;
        else
            if isfield(options, 'W'),
                options = rmfield(options, 'W');
            end
        end

        R = exp_acmmm(X, L, options);
        fprintf('dimension=%d, sigmak=%d, alpha=%d, beta=%d, f=%d\n', ...
            dimension(i), sigmak_(i), options.alpha, options.beta, R.f);

        if ~ use_cache,
            W = R.W;
            save(filename, 'W');
        end
    end
end