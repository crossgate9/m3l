%% init with 10% training data
% comment after running
drfun = 'MM_PCA';

% options = [];
% options.percent = 0.1;
% options.dimension = 1;
% options.knn = 1;
% options.drfun = drfun;
% 
% R = exp_acmmm(X, L, options);
% 
% train = R.train.raw;
% test = R.test.raw;
% 
% train_label = L(:, R.train.logical);
% test_label = L(:, R.test.logical);

%% exp
options.train = [];
options.train.raw = train;
options.train.label = train_label;
options.test = [];
options.test.raw = test;
options.test.label = test_label;

V = length(train);
dimension_ = [1 2 3 4 5 6 7 8 9 10];
knn_ = [1 2 3 4 5];

[dimension, knn] = meshgrid(dimension_, knn_);
n = numel(dimension);

fprintf('ratio 0.1 ... \n');
f = cell(4, 1);
f{4} = zeros(n, 1);
for i=1:n,
    options.dimension = dimension(i);
    options.knn = knn(i);
    R = exp_acmmm(X, L, options);
    % comment this for PCA
%     options.W = R.W;
    fprintf('dimension=%d, knn=%d, f1=%d\n', dimension(i), knn(i), R.f);
    f{4}(i) = R.f;
end
options = rmfield(options, 'W');

% adjust ratio, move half training to testing
% 10% -> 5% -> 2.5% -> 1.25%
train_half = train;
test_half = test;
train_half_label = train_label;
test_half_label = test_label;
ratio = 0.1;
for j=1:3,
    f{4-j} = zeros(n, 1);
    [train_logical, test_logical] = crossvalind('Holdout', train_half_label, 0.5);
    
    for k=1:V,
        test_half{k} = [test_half{k} train_half{k}(:, test_logical)];
        train_half{k} = train_half{k}(:, train_logical);
    end
    
    test_half_label = [test_half_label train_half_label(:, test_logical)];
    train_half_label = train_half_label(train_logical);
    
    options.train.raw = train_half;
    options.train.label = train_half_label;
    options.test.raw = test_half;
    options.test.label = test_half_label;
    
    ratio = ratio / 2;
    fprintf('ratio %d ... \n', ratio);
    for i=1:n,
        options.dimension = dimension(i);
        options.knn = knn(i);
        R = exp_acmmm(X, L, options);
        % comment this for PCA
%         options.W = R.W;
        fprintf('dimension=%d, knn=%d, f1=%d\n', dimension(i), knn(i), R.f);
        f{4-j}(i) = R.f;
    end
    options = rmfield(options, 'W');
end