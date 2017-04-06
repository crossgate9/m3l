function [R] = get_sub_w(W, d)

V = length(W);
R = cell(V, 1);

for i=1:V,
    R{i} = W{i}(:, 1:d);
end;

end