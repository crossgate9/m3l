function [n_train_features, n_test_features] = ...
    normalize_train_test(train_features, test_features)

options = [];
options.name = 'normalize_train_test';
[n_train_features, max_, min_] = normalize_(train_features, options);

options.max = max_;
options.min = min_;
[n_test_features, ~, ~] = normalize_(test_features, options);