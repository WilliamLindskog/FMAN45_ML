%% TASK 4

x = [-3, -2, -1, 0, 1, 2, 4];
y = [1, 1, -1, -1, -1, 1, 1]; 
x2 = linspace(-4, 5, 500);
y2 = (2/3)*x2.^2 - (5/3);

% PLOT TOGETHER
scatter(x,y)
hold on
plot(x2, y2);
hold off

%% TASK E1

% CLEAR
clear

% LOAD DATA
load A2_data

% COMPUTE LINEAR PRICIPAL COMPONENT ANALYSIS (PCA) AND VIZUALIZE
pi_use = linear_PCA(train_data_01, train_labels_01); 

%% TASK E2 PLOT FOR 2 CLUSTERS

[cluster_2_y, cluster_2_C] = K_means_clustering(train_data_01, 2);

gscatter(pi_use(1,:),pi_use(2,:), cluster_2_y, 'bk', 'xo')
set(gca,'FontSize',12)
title('2-DIMENSIONAL PCA')
xlabel('PC 1')
ylabel('PC 2')
lg = legend('Cluster 1', 'Cluster 2');
lg.FontSize = 10;

%% TASK E2 PLOT FOR 5 CLUSTERS

[cluster_5_y, cluster_5_C] = K_means_clustering(train_data_01, 5);

gscatter(pi_use(1,:),pi_use(2,:), cluster_5_y, 'bkrcg', 'xo+ph')
set(gca,'FontSize',12)
title('5-DIMENSIONAL PCA')
xlabel('PC 1')
ylabel('PC 2')
lg = legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5');
lg.FontSize = 10;

%% TASK E3 2 CLUSTERS CENTROIDS

% FORM IMAGES
img_C2a = reshape(cluster_2_C(:, 1), [28 28]);
img_C2b = reshape(cluster_2_C(:, 2), [28 28]);

% PLOT IMAGES USING imshow()
hold on
subplot(1,2,1);
imshow(img_C2a);
title('Cluster 1')
subplot(1,2,2);
imshow(img_C2b);
title('Cluster 2')
hold off

%% TASK E3 2 CLUSTERS CENTROIDS

% FORM IMAGES
img_C5a = reshape(cluster_5_C(:, 1), [28 28]);
img_C5b = reshape(cluster_5_C(:, 2), [28 28]);
img_C5c = reshape(cluster_5_C(:, 3), [28 28]);
img_C5d = reshape(cluster_5_C(:, 4), [28 28]);
img_C5e = reshape(cluster_5_C(:, 5), [28 28]);

% PLOT IMAGES USING imshow()
hold on
subplot(1,5,1);
imshow(img_C5a);
title('Cluster 1')
subplot(1,5,2);
imshow(img_C5b);
title('Cluster 2')
subplot(1,5,3);
imshow(img_C5c);
title('Cluster 3')
subplot(1,5,4);
imshow(img_C5d);
title('Cluster 4')
subplot(1,5,5);
imshow(img_C5e);
title('Cluster 5')
hold off

%% TASK E4, 2 CLUSTERS

clear

% LOAD DATA
load A2_data

% GET Y AND CENTROID FROM TRAINING DATA USING 2 CLUSTERS
[label_train, cluster_2_C] = K_means_clustering(train_data_01, 2);

% CHECK CLASSIFIER PERFORMANCE
[train_per, test_per] = K_means_classifier(label_train, train_labels_01, test_data_01, test_labels_01, cluster_2_C);

%% TASK E5, K CLUSTERS

clear

% LOAD DATA
load A2_data

% SET MAX CLUSTER VALUE AND LENGTH
K_max = 20;
len = length(test_labels_01);

% CALCULATE MISCLASSIFICATION RATES FOR VARIOUS K CLUSTERS
for i = 2:K_max
    
    % CLUSTERING
   [label_train, cluster_i_C] = K_means_clustering(train_data_01, i);
   
   % GET PERFORMANCE
   [train_i_per, test_i_per] = K_means_classifier(label_train, train_labels_01, test_data_01, test_labels_01, cluster_i_C);
   
   rate(i-1) = sum(test_i_per(:,4))*100 / len
   i
end

%% PLOT TASK E5
x_axis = 2:K_max;
plot(x_axis, rate, '.k', 'MarkerSize', 22)
title('Misclassification rate for increasing amount of clusters')
xlabel('Nbr Clusters')
ylabel('Misclassification rate (%)')

%% TASK E6 

clear 

% LOAD DATA
load A2_data

% GET SUPPORT VECTOR MACHINE 
train_data_transformed = train_data_01';
test_data_transformed = test_data_01';
svm_use = fitcsvm(train_data_transformed, train_labels_01);

% GET PREDICTION FOR TRAIN AND TEST DATA
train_prediction = predict(svm_use, train_data_transformed);
test_prediction = predict(svm_use, test_data_transformed);

% GET OVERALL SVM PERFORMANCE (TRAIN AND TEST)
svm_performance_train = svm_classification(train_prediction, train_labels_01);
svm_performance_test = svm_classification(test_prediction, test_labels_01);

%% TASK E7

clear

% LOAD DATA
load A2_data

% SET BETA (INCREASE WITH 0.5) 
beta = linspace(1,6,26)

% OPTIMAL BETA IS 4.72

%% LOOP TASK E7

for i = 1:length(beta)
    % GET SVM USING GAUSSIAN KERNEL
    gauss_svm = fitcsvm(train_data_01',train_labels_01,'KernelFunction','gaussian','KernelScale', beta(i));

    % GET PREDICTIONS FOR TRAINING AND TEST DATA
    train_prediction = predict(gauss_svm, train_data_01');
    test_prediction = predict(gauss_svm, test_data_01');

    % GET OVERALL SVM PERFORMANCE (TRAIN)
    svm_performance_train = svm_classification(train_prediction, train_labels_01);
    train_misclassification_rate = (svm_performance_train(2) + svm_performance_train(4)) / length(train_prediction)

    % GET OVERALL SVM PERFORMANCE (TRAIN)
    svm_performance_test = svm_classification(test_prediction, test_labels_01);
    test_misclassification_rate = (svm_performance_test(2) + svm_performance_test(4)) / length(test_prediction)
    
    if test_misclassification_rate == 0
        break
    end
end

%% TASK E7 EXTENDED TESTING

beta = 4.72;

% GET SVM USING GAUSSIAN KERNEL
gauss_svm = fitcsvm(train_data_01',train_labels_01,'KernelFunction','gaussian','KernelScale', beta);

% GET PREDICTIONS FOR TRAINING AND TEST DATA
train_prediction = predict(gauss_svm, train_data_01');
test_prediction = predict(gauss_svm, test_data_01');

% GET OVERALL SVM PERFORMANCE (TRAIN)
svm_performance_train = svm_classification(train_prediction, train_labels_01);
train_misclassification_rate = (svm_performance_train(2) + svm_performance_train(4)) / length(train_prediction)

% GET OVERALL SVM PERFORMANCE (TRAIN)
svm_performance_test = svm_classification(test_prediction, test_labels_01);
test_misclassification_rate = (svm_performance_test(2) + svm_performance_test(4)) / length(test_prediction)