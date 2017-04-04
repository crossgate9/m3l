function D = merge_views(X, L)

V = length(X);
n = size(L, 2);

% merge data into big matrix
d = 0;
for i=1:V,
    d = d + size(X{i}, 1);
end

j = 0;
D = zeros(n, d);
for i=1:V,
    D(:, j+1:j+size(X{i}, 1)) = X{i}';
end
