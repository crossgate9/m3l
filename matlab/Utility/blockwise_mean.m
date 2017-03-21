function r = blockwise_mean(A, block_size)
%BLOCKWISE Distince block processing for image with MEAN function.

floored_block_size = floor(block_size);
correct_size = isequal(size(floored_block_size), [1 2]);
non_negative = all(floored_block_size > 0);
non_inf = ~any(isinf(floored_block_size));

if ~(isnumeric(floored_block_size) && correct_size && non_negative && non_inf),
    error(message('Block size not valid.'));
end

if ~all(block_size == floored_block_size),
    warning(message('Block size should be integer.'));
end

fun = @(block_struct) mean(block_struct.data(:));

r = blockproc(A, floored_block_size, fun);

end