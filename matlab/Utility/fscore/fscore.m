function [f] = fscore(annotation, predicts, beta)
%FSCORE calcuate F-Score for statistical analysis of binary classification
%   annotation: vector of labels, 0 or 1 for elements
%   predicts: vector of predicted labels, 0 or 1 for elements
%   beta: beta in the general formulate
%   reference: https://en.wikipedia.org/wiki/F1_score

TP = sum(predicts(annotation == 1));
FP = sum(predicts(annotation == 0));
FN = sum(1 - predicts(annotation == 1));

precision = TP / (TP + FP);
recall = TP / (TP + FN);
beta2 = beta^2;
f = (1 + beta2) * (precision * recall) / (beta2 * precision + recall);

end

