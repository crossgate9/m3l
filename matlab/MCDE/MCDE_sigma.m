function sigma = MCDE_sigma(X, k, para)
% MCDE_SIGMA deduct optimized sigma for each view in MCDE
%   X: the V*1 or 1*V cell, where each element X_v corresponds to the D_v*n matrix of the
%   data from the v-th modality, D_v is the original dimension of the v-th modality,
%   and n is the number of data points.
%   k: K-th nearest neighbour of each data points.
%   para: parameters.
%   para.distance: function name to calculate distance. default: euclidean.
%   possible distance function, please refer to "help pdist2".

if ~isfield(para, 'distance'),
    para.distance = 'euclidean';
end

v = length(X);
n = size(X{1}, 2);

sigma = zeros(v, 1);
for i=1:v,
    D = pdist2(X{i}', X{i}', para.distance);
    D = sort(D, 2);
    if k+1 > size(D, 1),
        k = size(D, 1) - 1;
    end
    sigma(i) = mean(D(:, k+1) .^ 2);
end

end