function [predicts] = cKNN(train_features, train_label, test_features, k)

%% check parameters
if nargin < 3,
    error('not enough parameters');
end

if nargin == 3,
    k = 1;
end

train_sz = size(train_features);
test_sz = size(test_features);
train_label_length = length(train_label);

if train_label_length ~= train_sz(1),
    error('train data and label count not match');
end

if train_sz(2) ~= test_sz(2),
    error('train data and test data dimension not match');
end

%% KNN
MV = version('-release');

if strcmp(MV, '2009a') || strcmp(MV, '2013a'),
    predicts = knnclassify(test_features, train_features, train_label, k);
else
    mdl = fitcknn(train_features, train_label, 'NumNeighbors', k);
    predicts = mdl.predict(test_features);
end