function W = MM_PCA(X, L, para)

if ~isfield(para, 'ReducedDim')
    para.ReducedDim = 10;
end

D = merge_views(X, L);
[eigvector, ~] = PCA(D, para);
W = eigvector;

