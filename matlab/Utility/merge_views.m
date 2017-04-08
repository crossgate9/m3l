function D = merge_views(X, L)
    % X V*1 cell, for each cell, Dv*n
    % L: V*n

V = length(X);
n = size(L, 2);

if size(X{1}, 2) ~= n,
    for i=1:V,
        X{i} = X{i}';
    end
end

% merge data into big matrix
d = 0;
for i=1:V,
    d = d + size(X{i}, 1);
end

j = 0;
D = zeros(n, d);
for i=1:V,
    D(:, j+1:j+size(X{i}, 1)) = X{i}';
    j = j + size(X{i}, 1);
end
