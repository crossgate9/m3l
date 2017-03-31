percent_ = [0.1];
dimension_ = [1 2 3 4 5 6 7 8 9 10];
knn_ = [1 2 3 4 5];

[dimension, knn] = meshgrid(dimension_, knn_);

drfun = 'MCDE_KOL_2';

n = numel(dimension);

f = cell(100, 1);
for i=1:4,
    flag = true;
    options = [];
    options.percent = percent_(i);
    
    idx = options.percent * 100;
    f{idx} = zeros(n, 1);
    
    for j=1:n,
        options.dimension = dimension(j);
        options.knn = knn(j);
        options.drfun = drfun;
        options
        if flag,
            flag = false;
            R = exp_acmmm(X, L, options);
            f{i}(j) = R.f;
            R.f
            options.train = R.train;
            options.test = R.test;
            options.W = R.W;
            continue;
        end
        R = exp_acmmm(X, L, options);
        f{i}(j) = R.f;
        R.f
    end
end
