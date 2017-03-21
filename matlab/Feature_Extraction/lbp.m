function [R] = lbp(I)
%LBP Local Binary Pattern
%   R = LBP(I) extracts local binary pattern of the image I, 
%   with R = 1.0, P = 8.

%% init
% convert into grayscale if needed
if size(I, 3) == 3,
    I = rgb2gray(I);
end

% offset array for neighbourhood
p = 8;
dy = [-1 -1 -1 0 0 1 1 1];
dx = [-1 0 1 -1 1 -1 0 1];

%% main
[h, w] = size(I);

dy_min = min(dy);
dy_max = max(dy);
dx_min = min(dx);
dx_max = max(dx);

block_size_y = dy_max - dy_min + 1;
block_size_x = dx_max - dx_min + 1;

Cy = 1 - dy_min;
Cx = 1 - dx_min;
Ch = h - block_size_y;
Cw = w - block_size_x;
C = I(Cy:Cy+Ch, Cx:Cx+Cw);

R = zeros(Ch+1, Cw+1);
for i = 1:p,
   y = dy(i) + Cy;
   x = dx(i) + Cx;
   
   N = I(y:y+Ch, x:x+Cw);
   D = N >= C;
   v = 2^(i-1);
   R = R + v*D;
end

bins = 2^p;
R = hist(R(:), 0:(bins-1));

end