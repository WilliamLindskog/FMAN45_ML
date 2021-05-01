function [train_performance, test_performance] = K_means_classifier(train_labels, train_labels_given, test_data, test_labels_given, centroid)
% FUNCTION TAKES AS INPUT:
    % data - DATA (TRAINING, VALIDATION, OR TEST)
    % centroid 
% GIVES PERFORMANCE 
    
[~, n_use] = size(test_data);
[~, c_use] = size(centroid);

% INITIATE LABELS 
lbl_test = zeros(n_use,1);

for i = 1:n_use
    d = zeros(c_use, 1);
    for j = 1:c_use
        d(j) = fxdist(test_data(:,i),centroid(:,j));
    end
    lbl_test(i) = find(d == min(d));
end

[train_performance] = classification(train_labels, train_labels_given, centroid);
[test_performance] = classification(lbl_test, test_labels_given, centroid);

end

function d = fxdist(x,C)
% CALUCLATE DISTANCE AS IN PREVIOUS TASK 
    d = sqrt(sum((x-C).^2));
end

function [performance] = classification(labels_use, given_labels, centroid)
    % GET AMOUNT OF CENTROIDS AND LENGHT 
    [~, N] = size(centroid);
    
    % GET LABELS FOR CENTROIDS 
    c_lbl = zeros(N,1);
    for i = 1:N
        n_labels = given_labels(labels_use == i);
        c_lbl(i) = mode(n_labels);
    end
    
    % GET OVERALL PERFORMANCE  
    performance = zeros(N,4);
    
    % RIGHT LABEL (1), WRONG LABEL (2), ASSIGNED TO LABEL # (3) 
    for i = 1:N
        cluster = given_labels(labels_use == i);
        performance(i,1) = length(cluster) - sum(cluster);
        performance(i,2) = sum(cluster);
        performance(i,3) = c_lbl(i);
        
        % FOR MISCLASSIFICATION RATES
        if performance(i,3) == 0
            performance(i,4) = performance(i,2);
        else
            performance(i,4) = performance(i,1);
        end
    end
end