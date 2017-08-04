function [output, max_, min_] = normalize_(data, options)

%% initialization
if ~isfield(options, 'max'),
    max_ = max(data(:));
else
    max_ = options.max;
end

if ~isfield(options, 'min'),
    min_ = min(data(:));
else
    min_ = options.min;
end

%% main
output = (data - min_) / (max_ - min_);
