function pi_data = linear_PCA(data, labels)
% THIS FUNCTION TAKES DATA AND COMPUTES A LINEAR PCA
% WITH VISUALIZATION 

% ZERO MEAN DATA 
data = data - mean(data, 2); 

% SINGULAR VALUE DECOMPOSITION
[W_mat,diag_mat,V_mat] = svd(data);
W_d = W_mat(:,1:2);

% USE FOR VISUALIZATION
pi_data = W_d'*data;

% SCATTER PLOT
gscatter(pi_data(1,:),pi_data(2,:), labels, 'bk', 'xo')
set(gca,'FontSize',12)
title('2-DIMENSIONAL PCA')
xlabel('PC 1')
ylabel('PC 2')
lg = legend('0', '1');
lg.FontSize = 10;
end