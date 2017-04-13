function W = M3L(X,L,para)
% function W = M3L(X,L,para)
%
% X: the V*1 or 1*V cell, where each element X_v corresponds to the D_v*n matrix of the
% data from the v-th view, D_v is the original dimension of the v-th view,
% and n is the number of data points.
%
% L: the l*n label matrix, where l is the number of labels and n is the
% number of data points, each column represents the label vector of the
% correpsonding data point.
%
% para: parameters.
% para.sigma: the bandwidth of neighborhood graph matrix
% para.labelForm: the form of the label information, it could be either
% 'abs', 'inner', or 'Euclidean'
% para.alpha: the parameter to balance geometry and discrimination, the
% larger the alpha, the more important the geometry
%
% W: the V*1 cell, where each element W_v corresponds to the D_v*d transformation
% matrix of the original data from the v-th view.

%% initialization
V = length(X);
n = size(L,2);

if ~isfield(para,'sigma')
    para.sigma = ones(V, 1);
end

if ~isfield(para,'labelForm')
    para.labelForm = 'abs';
end

if ~isfield(para,'alpha')
    para.alpha = 0.5;
end
%% Geometry Preserving Criteria %%
N = cell(1, V); % Neighborhood Graph Matrix
S_l = cell(1, V); % Locality Matrix
S_g = cell(1, V); % Globality Matrix
for v = 1:V 
    NG = L2_distance(X{v}, X{v}, 1);
    NG = - (NG .* NG) / (2 * para.sigma(v));
    N{v} = exp(NG);

    S_l{v} = 2*X{v}*(diag(sum(N{v}))-N{v})*X{v}';
    S_g{v} = 2*X{v}*(diag(sum(1-N{v}))-(1-N{v}))*X{v}';
end
fprintf('Geometry Preserving Criteria ... done ... \n');

%% Discrimination Maximizing Criteria %%
A = zeros(n, n); % Similarity Matrix
% D = []; % Dissimilarity Matrix

if strcmp(para.labelForm, 'abs'),
    lf = @(L,i,j) (1-abs(L(:,i)-L(:,j)));
elseif strcmp(para.labelForm, 'inner'),
    lf = @(L,i,j) (L(:,i)/max(norm(L(:,i)),10^-5))'*(L(:,j)/max(norm(L(:,j)),10^-5));
elseif strcmp(para.labelForm, 'Euclidean'),
    lf = @(L,i,j) (1/max(L2_distance(L(:,i),L(:,j)),10^-5));
else
    error('Distance metric wrong!');
end

for i = 1:n
    for j = 1:n
        A(i,j) = lf(L, i, j);
    end
end
D = 1 - A;
fprintf('Discrimination Maximizing Criteria ... done ... \n');

%% Objective Function %%
S_L = []; % Big Locality Matrix
S_G = []; % Big Globality Matrix
for v = 1:V
    S_L = blkdiag(S_L,S_l{v}); % block diagonal matrix
    S_G = blkdiag(S_G,S_g{v}); % block diagonal matrix
end

Adj_A = kron(ones(V,V),A);
Adj_D = kron(ones(V,V),D);
L_A = diag(sum(Adj_A)) - Adj_A; % nV*nV Laplacian matrices
L_D = diag(sum(Adj_D)) - Adj_D; % nV*nV Laplacian matrices
L_A_cell = mat2cell(L_A, n*ones(1,V), n*ones(1,V)); % divide the big matrix into V*V blocks of size N*N
L_D_cell = mat2cell(L_D, n*ones(1,V), n*ones(1,V)); % divide the big matrix into V*V blocks of size N*N

Q_A_cell = cell(V, V);
Q_D_cell = cell(V, V);
for i = 1:V
    for j = 1:V
        Q_A_cell{i,j} = X{i}*L_A_cell{i,j}*X{j}';
        Q_D_cell{i,j} = X{i}*L_D_cell{i,j}*X{j}';
    end
end

Q_A = cell2mat(Q_A_cell); % Big Similarity Matrix
Q_D = cell2mat(Q_D_cell); % Big Dissimilarity Matrix

M_Max = para.alpha*S_G + (1-para.alpha)*Q_D; % scatter matrix to be maximized
M_Min = para.alpha*S_L + (1-para.alpha)*Q_A; % scatter matrix to be minimized

fprintf('Objective Function ... done ... \n');

%% Analytical Solution %%
[eigvector,eigvalue] = eig(M_Max,M_Min); %generalized eigendecomposition
[value index] = sort(diag(eigvalue),'descend'); % sort according to the scales of eigenvalues
eigvector_sort = eigvector(:,index); % rank the eigenvectors accroding to the eigenvalue sorting
W_matrix = eigvector_sort(:, value>0); % only select the eigenvectors with positive eigenvalues

% divide W_matrix to V matrices for different views
Dim_V = zeros(1, V);
for v = 1:V
    Dim_V(v) = size(X{v},1);
end
W = mat2cell(W_matrix, Dim_V); 

fprintf('Analytical Solution ... done ... \n');
