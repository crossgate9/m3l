function [hash] = split_data(X, L, options)

%% init
if ~ isfield(options, 'drfun'),
    error(message('Dimension Reduction function name is required'))
end

%% main
R = [];
expr = sprintf('R = exp_acmmm(X, L, options);');
eval(expr);

%% prepare output
hash = DataHash(R.train.logical);

if ~ exist('cache', 'dir'),
   mkdir('cache'); 
end

cache_folder = sprintf('cache/%s', hash);
if ~ exist(cache_folder, 'dir'),
    mkdir(cache_folder);
end

train_data_filename = sprintf('cache/%s/train.mat', hash);
test_data_filename = sprintf('cache/%s/test.mat', hash);

train_data = R.train.raw;
test_data = R.test.raw;
train_label = L(:, R.train.logical);
test_label = L(:, R.test.logical);
train_logical = R.train.logical;
test_logical = R.test.logical;

save(train_data_filename, 'train_data', 'train_label', 'train_logical');
save(test_data_filename, 'test_data', 'test_label', 'test_logical');

