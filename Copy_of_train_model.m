%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: your name
% date: date for release this code

clear; clc;

setup;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()
[data, wordMap] = read_data;

% get the cell and convert word from wordMap
input = data(:,2);
width = length(num2str(length(wordMap))-'0');
mapped_input = cellfun(@(s) sent2mat(s, wordMap, width), input, 'UniformOutput', false);

% get the target value
target = data(:,3);

%% Section 2: training
% Note: 
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions: 
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()


sentence_arr = cell2mat(mapped_input(i));

% reshape the array to matrix with width to num_of_digits
x = transpose(reshape(sentence_arr, width, []));

% set three w
w1 = rand(2, width, 1, 2, 'single');
w2 = rand(3, width, 1, 2, 'single');
w3 = rand(4, width, 1, 2, 'single');

% for each example in train.txt do
% section 2.1 forward propagation and compute the loss
for i = 1:10
    % convolution
    y1 = vl_nnconv(x, w1, []);
    y2 = vl_nnconv(x, w2, []);
    y3 = vl_nnconv(x, w3, []);

    % relu
    z1 = vl_nnrelu(y1); 
    z2 = vl_nnrelu(y2);
    z3 = vl_nnrelu(y3);

    % pooling
    y1_pool = vl_nnpool(z1, [size(z1,1), size(z1,2)]);
    y2_pool = vl_nnpool(z2, [size(z2,1), size(z2,2)]);
    y3_pool = vl_nnpool(z3, [size(z3,1), size(z3,2)]);

    % concat
    y_concat_cell = [y1_pool; y2_pool; y3_pool];
    
    % downsize to 2 classes for softmax
    y_concat_pool = vl_nnpool(y_concat_cell, ...
        [size(y_concat_cell,1), size(y_concat_cell,2)]);
    %y_concat = vl_nnconcat(y_concat_pool, 1);
    
    % softmax
    y_softmax = vl_nnsoftmax(y_concat_pool);
    
    % loss function
    loss = vl_nnloss(y_softmax, cell2mat(target(i)));
    
    display(loss)
end

% section 2.2 backward propagation and compute the derivatives
% TODO: your code


% section 2.3 update the parameters
% TODO: your code



