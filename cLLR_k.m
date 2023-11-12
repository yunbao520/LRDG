function [B,W,L] = cLLR_k(X,k)

[n, ~] = size(X);

% Neighborhood preserving regularizer with Heat kernel
%======= construct the affinity matrix of sample space ===========%
options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = k;
options.WeightMode = 'HeatKernel';
options.t = 1;
W = constructW(X, options);
B=diag(sum(W,2));
L = B-W;

end




