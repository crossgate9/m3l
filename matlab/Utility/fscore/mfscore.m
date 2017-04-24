function [f] = mfscore(annotation, predicts, options)
%FSCORE calcuate F-Score for statistical analysis of multi-label classification
%   annotation: vector of labels, starting from 0, consecutive
%   predicts: vector of predicted labels, starting from 0, consecutive
%   options: 
%       beta: beta in the general formulate
%       type: 'micro' or 'marco'
%   reference: https://en.wikipedia.org/wiki/F1_score
%              http://rushdishams.blogspot.co.id/2011/08/micro-and-macro-average-of-precision.html

%% init
if ~ isfield(options, 'beta'),
    options.beta = 1;
end

if ~ isfield(options, 'type'),
    options.type = 'micro';
end

%% main
C = max(annotation) + 1;
cm = zeros(C, C);

for i=1:C,
    for j=1:C,
        a = annotation == (j-1);
        p = predicts == (i-1);
        cm(i, j) = sum(a.*p);
    end
end

TP = zeros(C, 1);
FP = zeros(C, 1);
FN = zeros(C, 1);
P = zeros(C, 1);
R = zeros(C, 1);
for i=1:C,
    TP(i) = cm(i, i);
    FP(i) = sum(cm(:, i)) - TP(i);
    FN(i) = sum(cm(i, :)) - TP(i);
    P(i) = TP(i) / (TP(i) + FP(i));
    R(i) = TP(i) / (TP(i) + FN(i));
end

beta2 = options.beta ^ 2;

if strcmp(options.type, 'micro'), 
    % micro-average fscore
    precision = sum(TP) / (sum(TP) + sum(FP));
    recall = sum(TP) / (sum(TP) + sum(FN));
else
    % marco-average fscore
    precision = mean(P);
    recall = mean(R);
end

f = (1 + beta2) * (precision * recall) / (beta2 * precision + recall);

end

% True positive: diagonal position, cm(x, x).
% False positive: sum of column x (without main diagonal), sum(cm(:, x))-cm(x, x).
% False negative: sum of row x (without main diagonal), sum(cm(x, :), 2)-cm(x, x).

% 
% True positive (TP1)= 12
% False positive (FP1)=9
% False negative (FN1)=3
% 
% Then precision (P1) and recall (R1) will be 57.14 and 80
% 
% and for a different set of data, the system's
% 
% 
% True positive (TP2)= 50
% False positive (FP2)=23
% False negative (FN2)=9
% 
% Then precision (P2) and recall (R2) will be 68.49 and 84.75
% 
% Now, the average precision and recall of the system using the Micro-average method is
% 
% Micro-average of precision = (TP1+TP2)/(TP1+TP2+FP1+FP2) = (12+50)/(12+50+9+23) = 65.96
% Micro-average of recall = (TP1+TP2)/(TP1+TP2+FN1+FN2) = (12+50)/(12+50+3+9) = 83.78

% Macro-average precision = (P1+P2)/2 = (57.14+68.49)/2 = 62.82
% Macro-average recall = (R1+R2)/2 = (80+84.75)/2 = 82.25
