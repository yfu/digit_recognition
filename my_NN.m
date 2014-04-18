function a = test_example_NN
close all;
load ex4data1;
% word_to_arabic('one');

addpath(genpath('/Users/yfu/repo/DeepLearnToolbox'));
addpath('~/Dropbox/Courses/Machine Learning/project/data');

image_files = dir('~/Dropbox/Courses/Machine Learning/project/data/*.png');
n_files = length(image_files);
% the matrix to hold all the images
height = 20;
width = 20;
images = zeros(n_files, height * width);
% The first column is the image label; the 2nd column is the author;
% the 3rd column is the image number
image_labels = cell(n_files, 3);

for i = 1:n_files
    cur_image_filename = image_files(i).name;
    cur_image_filename(2:4);
    toks = regexp(cur_image_filename, '^(\w+)_(\w+)(\d+)', 'tokens');
    image_labels{i, 1} = word_to_arabic(toks{1}{2});
    image_labels{i, 2} = toks{1}{1}; image_labels{i, 3} = toks{1}{3};
    if image_labels{i, 1} == 0
        image_labels{i, 1} = 10;
    end
    cur_image = imread(cur_image_filename);
    % Complement the images
    cur_image = imcomplement(cur_image);
    images(i, :) = reshape(cur_image, height * width, 1);
end

% Make sure the labels are no problems.
% image_labels{1:16,1}
% Visualize the data
displayData(images(1:16,:));
% displayData(X(1:16,:))


%% Convert y to the needed format
y_mat = zeros(size(X, 1), 10);
my_ind = sub2ind(size(X), (1:size(y_mat))', y);
y_mat(my_ind) = 1;

%% Randomly choose some, say 500, images to be the training set.
% test_x_size = 500;
% test_x_ind = randperm(size(X, 1), test_x_size);
% test_x = X(test_x_ind, :);
% % Use the ramdom 500 as the training set
% test_y = y_mat(test_x_ind, :);
% 
% X(test_x_ind, :) = [];
% y_mat(test_x_ind, :) = [];
% train_x = X;
% train_y = y_mat;
% figure;
% displayData(X(1:25,:));

%% Use our own data as test set
train_x = X;
train_y = y_mat;

my_image_labels = cell2mat(image_labels(:,1));
test_x = images;
test_y = zeros(size(images, 1), 10);

% Recall that 0 is labelled as 10
my_ind = sub2ind(size(test_y), (1:size(images, 1))', my_image_labels);
test_y(my_ind) = 1;
test_x(1:5,:)
train_x(1:5, :)

my_max = max(max(train_x));
my_min = min(min(train_x));

test_x = (test_x / max(max(test_x))) * (my_max - my_min) + my_min;

% total = zscore([train_x; test_x]);
% train_x = total(1: size(train_x,1),:);
% test_x(test_x > 0.9 * max(max(test_x))) = my_max;
for i = 1:size(test_x, 1)
    sorted = sort(test_x(i,:));
    thres = sorted(350);
    temp = test_x(i, :);
    temp(temp>thres) = my_max;
    test_x(i,:) = temp;
end
% test_x(test_x > 0.3) = my_max;
% test_x = total((size(train_x, 1)+1):size(total,1), :);
% test_x = zscore(test_x);

displayData(train_x(900:1000, :));
figure;
displayData(test_x(1:100, :));
% figure;
% displayData(train_x(1:100, :));

figure;
hist(test_x(:));
figure;
hist(train_x(:));


input_size = 400;
hidden_size = 50;
output_size = 10;

% normalize
% [train_x, mu, sigma] = zscore(train_x);
% test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rand('state',0)
nn = nnsetup([input_size hidden_size output_size]);
nn.weightPenaltyL2 = 1e-3;
% nn.activation_function = 'sigm';
% nn.dropoutFraction = 0.1
opts.numepochs =  10;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

[nn, L] = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
er
% assert(er < 0.08, 'Too big error');
end






