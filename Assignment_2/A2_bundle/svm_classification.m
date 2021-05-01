function svm_performance = svm_classification(prediction, labels)
% FUNCTION THAT TAKES AS INPUT:
    % PREDICTIONS
    % LABELS
% RETURNS OVERALL PERFORMANCE 

% CLASSIFICATION ON DATA
img0 = labels(prediction == 0);
img1 = labels(prediction == 1);

% GET SVM PERFORMANCE 
svm_performance = zeros(1,4); 

% CHECK FOR CORRECT OF FALSE LABELED INSTANCES TRUE 0 (1), FALSE 0 (2),
% TRUE 1 (3), FALSE 1 (4)
svm_performance(1,1) = sum(img0 == 0);
svm_performance(1,2) = sum(img0 == 1);
svm_performance(1,3) = sum(img1 == 1);
svm_performance(1,4) = sum(img1 == 0);

end