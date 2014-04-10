function test_example_NN
load ex4data1

addpath(genpath('/Users/yfu/repo/DeepLearnToolbox'));
addpath('~/Dropbox/Courses/Machine Learning/project/data')

image_files = dir('~/Dropbox/Courses/Machine Learning/project/data/*.png');
n_files = length(image_files);
% the matrix to hold all the images
height = 20;
width = 20;
images = zeros(n_files, height * width);
% The first column is the image label; the 2nd column is the author;
% the 3rd column is the iamge number
image_labels = cell(n_files, 3);

for i = 1:n_files
    cur_image_filename = image_files(i).name;
    cur_image_filename(2:4);
    toks = regexp(cur_image_filename, '^(\w+)_(\w+)(\d+)', 'tokens');
    image_labels{i, 1} = toks{1}{2}; image_labels{i, 2} = toks{1}{1}; image_labels{i, 3} = toks{1}{3};
    cur_image = imread(cur_image_filename);
    images(i, :) = reshape(cur_image, height * width, 1);
end
% Visualize the data
displayData(images)

% Get 100 samples and visualize them

% Convert y to the needed format
y_mat = zeros(size(X, 1), 10);
my_ind = sub2ind(size(X), (1:size(y_mat))', y);
y_mat(my_ind) = 1;

% Randomly choose some, say 500, images to be the training set.
test_x_size = 500;
test_x_ind = randperm(size(X, 1), test_x_size);
test_x = X(test_x_ind, :);
test_y = y_mat(test_x_ind, :);

X(test_x_ind, :) = [];
y_mat(test_x_ind, :) = [];
train_x = X;
train_y = y_mat;

% displayData(X(myX, :));

% Set up the target variables
% Convert the format from a vector to a matrix

% train_x = double(X);
% test_x  = double(test_x)  / 255;
% test_y  = double(test_y);

input_size = 400;
hidden_size = 50;
output_size = 10;

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rand('state',0)
nn = nnsetup([input_size hidden_size output_size]);
nn.weightPenaltyL2 = 1e-3;
% nn.activation_function = 'sigm';
% nn.dropoutFraction = 0.1
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

[nn, L] = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
er
% assert(er < 0.08, 'Too big error');
