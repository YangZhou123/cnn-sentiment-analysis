%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: your name
% date: date for release this code

clear; clc;
% disable if want results to be the same
rng(1234);
setup;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()
[data, wordMap] = read_data;

% seperate the training and validation set
training_set = data(1:length(data) * 0.8,2:3);
validation_set = data(length(data)*0.8+ 1: end,2:3);

% add <PAD> and <UNK> to the wordMap
wordMap('<PAD>') = length(wordMap) + 1;
wordMap('<UNK>') = length(wordMap) + 1;

% get the target value
% target = data(:,3);
% init embedding
d = 5;
total_words = length(wordMap);
%random sample from normal distributiion
% with mean=0, variance=0.1
T = normrnd(0, 0.1, [total_words, d]);

% init filters
filter_size = [2, 3, 4];
n_filter = 2;

W_conv = cell(length(filter_size), 1);
B_conv = cell(length(filter_size), 1);

for i = 1: length(filter_size)
    %get filter size
    f = filter_size(i);
    % init W with: FW X FH X FC X K
    W_conv{i} = normrnd(0, 0.1, [f, d, 1, n_filter]);
    B_conv{i} = zeros(n_filter, 1);
end

% init output layer
total_filters = length(filter_size)*n_filter;
n_class = 2;
W_out = normrnd(0, 0.1, [total_filters, 1, 1, n_class]);
B_out = zeros(n_class, 1);

%% Section 2: training
% Note: 
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions: 
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss() 
% for each example in train.txt do
% section 2.1 forward propagation and compute the loss
eta_output = 0.05;
eta_conv = 0.05;
eta = 0.05;
epoch = 10;

for iter = 1:epoch
    % set total loss
    total_loss = [];
    num_right = 0;
    for example_idx = 1:length(training_set)      
            input = training_set{example_idx,1};
            word_indexs = [ ];
        for word_idx = 1: length(input)
            % if length of input < 4 size padding before with <PAD>
            if length(input) < 4
                % check how many need to be padded
                num_padding = 4 - length(input);

                for padding_idx = 1: num_padding
                    word_indexs = [word_indexs, wordMap('<PAD>')];
                end
            end

            % check key exists
            % if not exists
            if isKey(wordMap, input{word_idx}) == 0
                word_indexs = [word_indexs, wordMap('<UNK>')];
            else
                word_indexs = [word_indexs, wordMap(input{word_idx})];
            end
        end

        X = T(word_indexs, :);
        pool_res = cell(1, length(filter_size));
        cache = cell(2, length(filter_size));

        for i = 1: length(filter_size)
            % convolutional operation
            conv = vl_nnconv(X, W_conv{i}, B_conv{i});

            % apply activation fuction: relu
            relu = vl_nnrelu(conv);

            % 1-max pooling operation
            sizes = size(conv);
            pool = vl_nnpool(relu, [sizes(1), 1]);

            % important: keep these values for back-prop
            cache{2, i} = relu;
            cache{1, i} = conv;
            pool_res{i} = pool;
        end


        % concatenate 
        % z = vl_nnconcat(pool_res, 3);
        z = squeeze(vl_nnconcat(pool_res, 3));
        % compute loss
        % o: value of output layer
        o = vl_nnconv(z, W_out, B_out);
        % y: ground truth label (1 or 2)
        
        if o(:,:,1) > o(:,:,2)
            predict = 1;
        else
            predict = 0;  
        end
        
        y = training_set{example_idx,2};
        
        if predict == y
            num_right = num_right + 1;
        end

        if (y == 0)
                y = 2;
        else
            y = 1;
        end
        loss = vl_nnloss(o, y);
        % append total loss
        total_loss = [total_loss, loss];

        DlossDo = vl_nnloss(o, y, 1);
        [DoDz, DoDW_out, DoDB_out] = vl_nnconv(z, W_out, B_out, DlossDo);
        %update parameter for output layer
        W_out = W_out - eta_output * DoDW_out;
        B_out = B_out - eta_output * DoDB_out;

        % backward concat
        dzinputs = cell(1, length(filter_size));

        % seperate the derrdz to three cell each contains the pool results.
        reshaped_DoDz = reshape(DoDz, [n_filter, length(filter_size)]);
        for i = 1:length(filter_size)
            for inner_idx = 1:n_filter
                dzinputs{i}(:,:,inner_idx) = reshaped_DoDz(inner_idx,i);
            end
        end

        % calculate for each filter
        for i = 1:length(filter_size)
            dz = dzinputs{i};

            % back nnpool
            relu_size = size(cache{2, i});
            dzdpool = vl_nnpool(cache{2, i}, [relu_size(1), 1], dz);

            % back relu
            dzdrelu = vl_nnrelu(cache{1, i}, dzdpool);

            % back to X
            [dzdx, dzdwconv, dzbconv] = vl_nnconv(X, W_conv{i}, B_conv{i}, dzdrelu);

            % section 2.3 update the parameters
            % update W_conv , B_conv parameters
            W_conv{i} = W_conv{i} - eta_conv * dzdwconv;
            B_conv{i} = B_conv{i} - eta_conv * dzbconv;
%             ???? how to update T
            for j = 1:length(word_indexs)
            T(word_indexs(j),:) = T(word_indexs(j),:) - eta * dzdx(j,:);
            end
        end

%         for j = 1:length(word_indexs)
%             T(word_indexs(j),:) = T(word_indexs(j),:) - eta * dzdx(j,:);
%         end
%         for i = 1:length(T)
%             T(i,:) = T(i,:) - eta * dzdx(i,:);
%         end
    end
    % append loss and reset total loss
    training_loss =  mean(total_loss);
    display(training_loss);
    total_loss = [];
    train_accuracy = num_right/length(training_set);
    display(train_accuracy);
    
    num_right = 0;
    for example_idx = 1:length(validation_set)
            input = validation_set{example_idx,1};
            word_indexs = [ ];
        for word_idx = 1: length(input)
            % if length of input < 4 size padding before with <PAD>
            if length(input) < 4
                % check how many need to be padded
                num_padding = 4 - length(input);

                for padding_idx = 1: num_padding
                    word_indexs = [word_indexs, wordMap('<PAD>')];
                end
            end

            % check key exists
            % if not exists
            if isKey(wordMap, input{word_idx}) == 0
                word_indexs = [word_indexs, wordMap('<UNK>')];
            else
                word_indexs = [word_indexs, wordMap(input{word_idx})];
            end
        end

        X = T(word_indexs, :);
        pool_res = cell(1, length(filter_size));
%         cache = cell(2, length(filter_size));

        for i = 1: length(filter_size)
            % convolutional operation
            conv = vl_nnconv(X, W_conv{i}, B_conv{i});

            % apply activation fuction: relu
            relu = vl_nnrelu(conv);

            % 1-max pooling operation
            sizes = size(conv);
            pool = vl_nnpool(relu, [sizes(1), 1]);

            % important: keep these values for back-prop
%             cache{2, i} = relu;
%             cache{1, i} = conv;
            pool_res{i} = pool;
        end


        % concatenate 
        % z = vl_nnconcat(pool_res, 3);
        z = squeeze(vl_nnconcat(pool_res, 3));
        % compute loss
        % o: value of output layer
        o = vl_nnconv(z, W_out, B_out);
        % y: ground truth label (1 or 2)
        
        if o(:,:,1) > o(:,:,2)
            predict = 1;
        else
            predict = 0;  
        end
        
        y = validation_set{example_idx,2};
        
        if predict == y
            num_right = num_right + 1;
        end

        if (y == 0)
                y = 2;
        else
            y = 1;
        end
        loss = vl_nnloss(o, y);
        % append total loss
        total_loss = [total_loss, loss];
    end

    validation_loss =  mean(total_loss);
    display(validation_loss);
    total_loss = [];
    display(num_right)
    validation_accuracy = num_right/length(validation_set);
    display(validation_accuracy);
    
    
    
end


% section 2.2 backward propagation and compute the derivatives
% TODO: your code



% section 2.3 update the parameters
% TODO: your code



