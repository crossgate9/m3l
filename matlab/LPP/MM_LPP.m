function W = MM_LPP(X, L, para)

if ~isfield(para, 'ReducedDim')
    para.ReducedDim = 10;
end

D = merge_views(X, L);
w = constructW(D, para);
[W, ~] = LPP(w, para, D);