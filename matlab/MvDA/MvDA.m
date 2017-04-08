function W = MvDA(X,L,para)
% function W = MvDA(X,L,para)
% Multi-view Discriminant Analysis
%
% X: the V*1 or 1*V cell, where each element X_v corresponds to the D_v*n matrix of the
% data from the v-th modality, D_v is the original dimension of the v-th modality,
% and n is the number of data points.
%
% L: the l*n label matrix, where l is the number of labels and n is the
% number of data points, each column represents the label vector of the
% correpsonding data point.
%
% para: parameters.
% para.beta: the parameters to balance the discrimination and modality-consistency.
%
% W: the V*1 cell, where each element W_v corresponds to the D_v*d transformation
% matrix of the original data from the v-th modality.

%% initialization

if ~isfield(para,'beta')
    para.beta = 0.1;
end

V = length(X);
n = size(L,2);

%% Discrimination Preserving %%
A = []; % Similarity Matrix
D = []; % Dissimilarity Matrix
for i = 1:n
    for j = 1:n
        A(i,j) = (1 - abs(L(:,i)-L(:,j)));
        D(i,j) = abs(L(:,i)-L(:,j));
    end
end

%% Modality Consistency Preserving %%
C = cell(1, V);
for v = 1:V 
    C{v} = pinv(X{v}'*X{v})*X{v}';
end

%% Objective Function %%

Adj_A = kron(ones(V,V),A);
Adj_D = kron(ones(V,V),D);
L_A = diag(sum(Adj_A)) - Adj_A; % nV*nV Laplacian matrices
L_D = diag(sum(Adj_D)) - Adj_D; % nV*nV Laplacian matrices
L_A_cell = mat2cell(L_A, n*ones(1,V), n*ones(1,V)); % divide the big matrix into V*V blocks of size N*N
L_D_cell = mat2cell(L_D, n*ones(1,V), n*ones(1,V)); % divide the big matrix into V*V blocks of size N*N

Q_A_cell = {};
Q_D_cell = {};
for i = 1:V
    for j = 1:V
        Q_A_cell{i,j} = X{i}*L_A_cell{i,j}*X{j}';
        Q_D_cell{i,j} = X{i}*L_D_cell{i,j}*X{j}';
    end
end

Q_A = cell2mat(Q_A_cell); % Big Similarity Matrix
Q_D = cell2mat(Q_D_cell); % Big Dissimilarity Matrix

% Big Consistency Matrix
C_diag = []; 
for v = 1:V
    C_diag = blkdiag(C_diag,C{v}'*C{v}); % block diagonal matrix
end
C_big = V*C_diag - cell2mat(C)'*cell2mat(C);

% matrix construction for generalized eigendecomposition
M_Max = Q_D; % scatter matrix to be maximized
M_Min = Q_A + para.beta*C_big; % scatter matrix to be minimized

%% Analytical Solution %%
[eigvector,eigvalue] = eig(M_Max,M_Min); %generalized eigendecomposition
[value index] = sort(diag(eigvalue),'descend'); % sort according to the scales of eigenvalues
eigvector_sort = eigvector(:,index); % rank the eigenvectors accroding to the eigenvalue sorting
W_matrix = eigvector_sort(:, value>0); % only select the eigenvectors with positive eigenvalues

% divide W_matrix to V matrices for different views
Dim_V = [];
for v = 1:V
    Dim_V = [Dim_V size(X{v},1)];
end
W = mat2cell(W_matrix, Dim_V); 
