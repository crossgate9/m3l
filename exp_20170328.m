percent_ = [0.01 0.02 0.05 0.1];
dimension_ = [1 2 3 4 5 6 7 8 9 10];
knn_ = [1 2 3 4 5];

[dimension, knn] = meshgrid(dimension_, knn_);

drfun = 'MCDE_KOL_2';

n = numel(dimension);
for i=1:4,
    flag = true;
    options = [];
    options.percent = percent_(i);
    
    for j=1:n,
        options.dimension = dimension(i);
        options.knn = dimension(i);
        options.drfun = drfun;
        options
        if flag,
            flag = false;
            R = exp_acmmm(X, L, options);
            options.train = R.train.raw;
            options.test = R.test.raw;
            continue;
        end
        R = exp_acmmm(X, L, options);
        R.f
    end
end
