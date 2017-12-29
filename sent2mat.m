function [ val ] = sent2mat( sentence, wordMap, num_of_digits)
%WORD_TO_VECTOR Summary of this function goes here
%   Detailed explanation goes here

    % convert word to index in wordMap
    index = cellfun(@(w) wordMap(w), sentence, 'UniformOutput', false);
    
    % convert index to vector by seperating the digits
    index_vec = cellfun(@(idx) num2str(idx)-'0', index, 'UniformOutput', false);
    
    % padding with index vector
    padded_index_vec = cellfun(@(idx) padarray(idx, [0 (num_of_digits-length(idx))], 0, 'pre'), ...
    index_vec, 'UniformOutput', false);

    val = single(cell2mat(padded_index_vec));
end

