function c = colorhist(I, n)
%% COLORHIST calculate histogram from an image object.
% c = COLORHIST(I, n) takes a grayscale or color image I, 
% and calculate its color histogram with N bins for image I
% above a grayscale colorbar of length N.
% this function can be considered as a wrapper for imhist.

%% main
% convert into grayscale if needed
if size(I, 3) == 3,
    I = rgb2gray(I);
end

c = imhist(I, n);


end