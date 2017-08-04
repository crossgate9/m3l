function [output] = accuracy(annotation, predicts)

if size(annotation, 1) == 1
    annotation = annotation';
end

if size(predicts, 1) == 1
    predicts = predicts';
end

if size(annotation, 1) ~= size(predicts, 1)
    error('annotation and predicts should have same length');
end

N = length(annotation);
output = sum(annotation == predicts) / N;

end

