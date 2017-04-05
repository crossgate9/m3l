function [R] = normalize_mod(W)
    if iscell(W),
        V = length(W);
        R = cell(V, 1);
        for j =1:V,
            R{j} = normalize_matrix_mod(W{j});
        end
    else
        R = normalize_matrix_mod(W);
    end
end


function [R] = normalize_matrix_mod(W)
    [r, c] = size(W);
    W = abs(W);
    R = zeros(r, c);
    for i = 1:c,
        m = sqrt(sum(W(:, i) .* W(:, i)));
        R(:, i) = W(:, i) / m;
    end
end