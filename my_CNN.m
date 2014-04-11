function my_CNN

addpath(genpath('/Users/yfu/repo/DeepLearnToolbox'));
addpath('~/Dropbox/Courses/Machine Learning/project/data')

% load mnist_uint8;
% % [train_x, mu, sigma] = zscore(double(train_x) / 255);
% % test_x = normalize(double(test_x) / 255, mu, sigma);
% train_x = double(reshape(train_x', 28, 28, 60000));
% test_x = double(reshape(test_x', 28, 28, 10000));
% train_y = double(train_y');
% test_y = double(test_y');

load ex4data1
rand('state',sum(100.*clock));
% Convert y to the needed format
y_mat = zeros(size(X, 1), 10);
my_ind = sub2ind(size(X), (1:size(y_mat))', y);
y_mat(my_ind) = 1;

% Randomly choose some, say 500, images to be the training set.
% Test set
test_x_size = 500;
test_x_ind = randperm(size(X, 1), test_x_size);
test_x_ind(1:5)
test_x = X(test_x_ind, :) * 500;
test_y = y_mat(test_x_ind, :);
test_y = double(test_y');

% training set
X(test_x_ind, :) = [];
y_mat(test_x_ind, :) = [];
train_x = X * 500;
% [train_x, mu, sigma] = zscore(train_x);
% test_x = normalize(test_x, mu, sigma);
% displayData(train_x(1:16,:))
train_y = y_mat;
train_y = double(train_y');

% Convert them to the format that CNN accepts
train_x = double(reshape(train_x',20,20,4500));
test_x = double(reshape(test_x',20,20,500));

% displayData(X(myX, :));

% Set up the target variables
% Convert the format from a vector to a matrix

% train_x = double(X);
% test_x  = double(test_x)  / 255;
% test_y  = double(test_y);

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 5;
opts.batchsize = 100;
opts.numepochs = 5;

cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);
er
%plot mean squared error
figure; plot(cnn.rL);

% assert(er<0.12, 'Too big error');