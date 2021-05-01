function [y,C] = K_means_clustering(X,K)

% Calculating cluster centroids and cluster assignments for:
% Input:    X   DxN matrix of input data
%           K   Number of clusters
%
% Output:   y   Nx1 vector of cluster assignments
%           C   DxK matrix of cluster centroids

[D,N] = size(X);

intermax = 50;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
y = zeros(N,1);
Cold = C;

for kiter = 1:intermax
    % Step 1: Assign to clusters
    for i = 1:N
        y(i) = step_assign_cluster(X(:,i), C, K);
    end
    
    % Step 2: Assign new clusters
    
    % CREATE NEW ARRAYS
    [C_use, diff_use] = step_compute_mean(X, y, C, K);
        
    if diff_use < conv_tol
        return
    end
    Cold = C_use;
    C = C_use;
end
end

function d = fxdist(x,C)
    d = sqrt(sum((x-C).^2));
end

function d = fcdist(C1,C2)
    d = sqrt(sum((C1-C2).^2));
end

function y = step_assign_cluster(X, C, K)
    dist = zeros(K,1);
    for i = 1:K
        dist(i) = fxdist(X, C(:, i));
    end
    y = find(dist == min(dist));
end

function [C_use, diff_use] = step_compute_mean(X, y, C, K)
    C_use = zeros(size(C));
    diff_use = zeros(K,1);
    
    for i = 1:K
        C_use(:,i) = mean(X(:,y == i), 2);
        diff_use(i) = fcdist(C_use(:,i), C(:,i));
    end
end