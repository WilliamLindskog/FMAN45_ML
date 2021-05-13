%% EXERCISE 2 

% TEST FULLY CONNECTED
test_fully_connected

%% EXERCISE 3

% TEST ReLU
test_relu

%% EXERCISE 4 (SOFTMAX)

% TEST SOFMAX (FORWARD)
test_softmaxloss

%% EXERCISE 4 (GRADIENT NET)

% TEST WHOLE GRADIENT NET
test_gradient_whole_net

%% EXERCISE 5

% NOT SURE WHAT TO TEST HERE

%% EXERCISE 6 (CHECK)

% TEST MNIST-STARTER (REACHES 97.58%) 
mnist_starter

%% EXERCISE 6 

% INDECIES OF 16 FIRST MISCLASSIFIED DIGITS

% ADD PATH 
addpath(genpath('./'));

% GET TEST SETS (FOLLOWING STEPS ARE MAINLY COPIED FROM mnist_starter.m
img_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
lbl_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');

% RESHAPE TEST SETS AND CREAT CLASSES
img_test = reshape(img_test, [28, 28, 1, 10000]);
lbl_test(lbl_test==0) = 10;

classes = [1:9 0];

% CREATE PREDICTION ARRAY AND BATCH SIZE (SET LABEL 0 TO 10 TO AVOID
% COLLISION USING zeros()
predictions = zeros(numel(lbl_test),1);
batch_size = 16;

% GET PREDICTIONS
for i = 1:batch_size:size(lbl_test)
    idx = i:min(i+batch_size-1, numel(lbl_test));
    % note that y_test is only used for the loss and not the prediction
    lbl = evaluate(net, img_test(:,:,:,idx), lbl_test(idx));
    [~, p] = max(lbl{end-1}, [], 1);
    predictions(idx) = p;
end

missclass_images = find(predictions ~= lbl_test, 16);

%% EXERCISE 6 PLOT MISCLASSIFIED IMAGES

batch_size = 9; 

for i=1:batch_size
    subplot(sqrt(batch_size),sqrt(batch_size),i);
    imagesc(img_test(:,:,:,missclass_images(i)));
    title(['Prediction Label: ', num2str(classes(predictions(missclass_images(i)))),newline, 'True Label: ', num2str(classes(lbl_test(missclass_images(i))))]);
    axis off;
end

%% Task 6 Confusion matrix

% TARGET AND OUTPUT LABELS
target_label = zeros(10,10000);
output_label = zeros(10,10000);

% GET INDICIES 
target_index = sub2ind(size(target_label), lbl_test', 1:10000);
output_index = sub2ind(size(output_label), predictions', 1:10000);
target_label(target_index) = 1;
output_label(output_index) = 1;

% PLOT CONFUSION MATRIX
plotconfusion(target_label,output_label)

%% Task 6 plot filters

% GET TRAINING INSTANCES 
img_train = loadMNISTImages('data/mnist/train-images.idx3-ubyte');
lbl_train = loadMNISTLabels('data/mnist/train-labels.idx1-ubyte');

% RESHAPE AS IN PREVIOUS TASK 
img_train = reshape(img_train, [28, 28, 1, 60000]);
lbl_train(lbl_train==0) = 10;

% SUBTRACT MEAN 
data_mean = mean(img_train(:));
img_train = bsxfun(@minus, img_train, data_mean);

% CONVOLUTIONAL LAYER 
first_conv_layer = cell2mat(net.layers(2));

% CONVOLUTIONAL LAYER PARAMETERS, WEIGHTS AND BIASES
first_conv_layer_params = first_conv_layer(1).params;
first_conv_layer_weights = first_conv_layer_params(1).weights;
first_conv_layer_bias = first_conv_layer_params(1).biases;

% CONVOLUTIONAL WITH PADDING 
y = conv_with_padding_forward(img_train(:,:,1,2), first_conv_layer_weights, first_conv_layer_bias, [2 2]);

% PLOTS 
batch_size = 16;
for i = 1:batch_size
    subplot(sqrt(batch_size),sqrt(batch_size),i)
    imshow(y(:,:,i))
end

%% EXERCISE 7 (TAKES LONG) (DON'T DO UNNECESSARILY)

% TEST BASELINE
cifar10_starter

%% Exercise 7 Misclassified 


load models/cifar10_Version4 net

addpath(genpath('./'));

[img_train, label_train, img_test, label_test, classes] = load_cifar10(4);

data_mean = mean(mean(mean(img_train, 1), 2), 4); % mean RGB triplet
img_test = bsxfun(@minus, img_test, data_mean);

pred = zeros(numel(label_test),1);
batch = 16;

for i=1:batch:size(label_test)
    i
    idx = i:min(i+batch-1, numel(label_test));
    % note that y_test is only used for the loss and not the prediction
    y = evaluate(net, img_test(:,:,:,idx), label_test(idx));
    [~, p] = max(y{end-1}, [], 1);
    pred(idx) = p;
end

misclassified = pred ~= label_test;

miss_images = find(misclassified, 32);
%% Task 7 Plotting misclassified images

batch_size = 16;
for i=1:batch_size
    subplot(sqrt(batch_size),sqrt(batch_size),i);
    imagesc((img_test(:,:,:,miss_images(i+16))+data_mean)/255);
    title(['Predicted Label: ', classes(pred(miss_images(i+16))), 'True Label: ', classes(label_test(miss_images(i+16)))]);
    axis off;
end

%% Task 7 Confusion matrix

targets = zeros(10,10000);
outputs = zeros(10,10000);

target_index = sub2ind(size(targets), label_test', 1:10000);
output_index = sub2ind(size(outputs), pred', 1:10000);
targets(target_index) = 1;
outputs(output_index) = 1;

plotconfusion(targets,outputs)

xlabel('Target Class','FontWeight','bold')
set(gca,'xticklabel',[classes; ' '])
ylabel('Output Class','FontWeight','bold')
set(gca,'yticklabel',[classes; ' '])

%% Task 7 plot filter

% SUBTRACT MEAN 
data_mean = mean(img_train(:));
img_train = bsxfun(@minus, img_train, data_mean);

% FIRST CONVOLUTIONAL LAYER
first_conv = cell2mat(net.layers(2));

% COMPUTE PARAMETERS, WEIGTHS AND BIASES
first_conv_params = first_conv(1).params;
first_conv_weights = first_conv_params(1).weights;
first_conv_bias = first_conv_params(1).biases;

% CONVOLUTIONAL WITH PADDING 
y = conv_with_padding_forward(img_train(:,:,:,2), first_conv_weights, first_conv_bias, [2 2]);

% SHOW IMAGES 
imshow((img_train(:,:,:,2)+data_mean)/255)
figure()
for i = 1:32
    subplot(8,4,i)
    imagesc(y(:,:,i)/255)
    axis off
end