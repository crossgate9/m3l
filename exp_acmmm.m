function [R] = exp_acmmm(X, L, para)

%% init
n = size(L, 2);

if ~isfield(para, 'percent')
    para.percent = 0.01;
end

if para.percent <= 0 || para.percent >= 1,
    error(message('Domain of percent should be (0, 1)'));
end

if ~isfield(para, 'dimension')
    para.dimension = 10;
end
para.ReducedDim = para.dimension;

if ~isfield(para, 'knn'),
    para.knn = 1;
end

if ~isfield(para, 'drfun'),
    para.drfun = 'MCDE_KOL_2';
end

if exist(para.drfun) == 0,
    error(message('Dimension Reduction Function not existed'));
end

R = [];

%% feature extraction
if isfield(para, 'verbose'),
    fprintf('feature extraction ... \n');
end

if isfield(para, 'train') && isfield(para, 'test'),
    % load existed data
    train = para.train.raw;
    test = para.test.raw;
    train_label = para.train.label;
    test_label = para.test.label;
    if isfield(para.train, 'logical') && isfield(para.test, 'logical'),
        train_logical = para.train.logical;
        test_logical = para.test.logical;
    else
        train_logical = [];
        test_logical = [];
    end
    V = length(train);
else
    % re-construct data
    V = 0;
    D = cell(1, 1);
    bow = zeros(1, 1);
    X_l = length(X);
    for i=1:X_l,
        if iscell(X{i}),
            % image feature

            % colorhist
            V = V + 1;
            bow(V) = 0;
            D{V} = zeros(n, 128);
            for j=1:n,
                D{V}(j, :) = colorhist(X{i}{j}, 128);
            end

            % lbp
            V = V + 1;
            bow(V) = 0;
            D{V} = zeros(n, 256);
            for j=1:n,
                D{V}(j, :) = lbp(X{i}{j});
            end

            % hog
            V = V + 1;
            bow(V) = 1;
            D{V} = cell(n, 1);
            for j=1:n,
                D{V}{j} = vl_hog(single(X{i}{j}), 10);
            end
        else
            % plain feature
            V = V + 1;
            D{V} = X{i};
        end
    end

    %% split dataset and perform bag of word
    if ~isfield(para, 'train_logical') || ~isfield(para, 'test_logical'),
        [train_logical, test_logical] = crossvalind('Holdout', L, 1-para.percent);
    else
        train_logical = para.train_logical;
        test_logical = para.test_logical;
    end
    
    train_count = sum(train_logical);
    test_count = sum(test_logical);

    train = cell(V, 1);
    test = cell(V, 1);
    for i=1:V,
        if bow(i) ~= 1,
            train{i} = D{i}(train_logical, :);
            test{i} = D{i}(test_logical, :);
            train{i} = train{i}';
            test{i} = test{i}';
            continue;
        end

        % bag of word
        train_ = D{i}(train_logical);
        test_ = D{i}(test_logical);

        sz = size(train_{1});
        s = sz(1) * sz(2);
        words = zeros(s * train_count, sz(3));
        for j=1:train_count,
            tmp = train_{j};
            tmp = shiftdim(tmp, 2);
            tmp = tmp(:, :);
            tmp = tmp';
            b = (j-1) * s + 1;
            sz_ = size(tmp);
            words(b:b+sz_(1)-1, 1:sz_(2)) = tmp;
        end

        words = words';
        [C, A] = vl_kmeans(words, 256);
        train{i} = zeros(train_count, s);

        for j=1:train_count,
            b = (j-1) * s + 1;
            a = j * s;
            train{i}(j, :) = A(b:a);
        end

        C = single(C);
        test{i} = zeros(test_count, s);
        for j=1:test_count,
            tmp = test_{j};
            tmp = shiftdim(tmp, 2);
            tmp = tmp(:, :);
            tmp = tmp';
            d = pdist2(tmp, C');
            [~, f] = sort(d, 2);
            test{i}(j, 1:numel(f(:, 1))) = f(:, 1);
        end
        
        train{i} = train{i}';
        test{i} = test{i}';
    end
    
    train_label = L(:, train_logical);
    test_label = L(:, test_logical);
end
%% main

if isfield(para, 'verbose'),
    fprintf('data normalization ... \n');
end

% transport & normalization
train_normalized = cell(V, 1);
test_normalized = cell(V, 1);
for i = 1:V,
    [train_normalized{i}, test_normalized{i}] = ...
        normalize_train_test(train{i}, test{i});
%     train_normalized{i} = train_normalized{i}';
%     test_normalized{i} = test_normalized{i}';
end

if isfield(para, 'verbose'),
    fprintf('dimension reduction & normalization ... \n');
end

if isfield(para, 'W'),
    W = para.W;
else
    options = [];
    options.distance = 'euclidean';
    options.sigma = MCDE_sigma(train_normalized, 10, options);
    % for PCA or LPP
    options.ReducedDim = para.ReducedDim;

    % reduced weight matrix
    W = [];
    expr = sprintf('W = %s(train_normalized, train_label, options);', para.drfun);
    eval(expr);
end

if iscell(W),
    % multi view weight matrix
    train_reduced = cell(V, 1);
    test_reduced = cell(V, 1);
    train_reduced_normalized = cell(V, 1);
    test_reduced_normalized = cell(V, 1);
    for i = 1:V,
        train_reduced{i} = train_normalized{i}' * W{i}(:, 1:para.dimension);
        test_reduced{i} = test_normalized{i}' * W{i}(:, 1:para.dimension);
        [train_reduced_normalized{i}, test_reduced_normalized{i}] = ...
            normalize_train_test(train_reduced{i}, test_reduced{i});
    end

    train_data = train_reduced_normalized{1};
    test_data = test_reduced_normalized{1};
    for i = 2:V,
        train_data = [train_data train_reduced_normalized{i}];
        test_data = [test_data test_reduced_normalized{i}];
    end
else
    % pca-like weight matrix
    train_data = merge_views(train_normalized, train_label);
    test_data = merge_views(test_normalized, test_label);
    train_reduced = train_data * W;
    test_reduced = test_data * W;
    [train_data, test_data] = normalize_train_test(train_reduced, test_reduced);
end

% k-NN
if isfield(para, 'verbose'),
    fprintf('knn ... \n');
end

mdl = fitcknn(train_data, train_label, 'NumNeighbors', para.knn);
predicts = mdl.predict(test_data);

f = fscore(test_label, predicts, 1);

if isfield(para, 'verbose'),
    fprintf('output ... \n');
end

R.f = f;
R.annotations = test_label;
R.predicts = predicts;

R.train = [];
R.train.logical = train_logical;
R.train.raw = train;
R.train.data = train_data;
R.train.label = train_label;

R.test = [];
R.test.logical = test_logical;
R.test.raw = test;
R.test.data = test_data;
R.test.label = test_label;

R.W = W;

R.percent = para.percent;
R.knn = para.knn;
R.drfun = para.drfun;