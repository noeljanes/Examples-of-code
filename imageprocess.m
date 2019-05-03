%% Header 
% This file imports an image, greyscales it and plots the surface
%%

a = rgb2gray(imread('IMG_9284.cr2')) ;                                     % Importing and greyscaling the image
%a = imrotate(a,10) ;                                                      % Rotating the image to align the center beam with the horizontal
[M , I] = max(a(:)) ;            
[I_row , I_col] = ind2sub(size(a),I) ;


offset1 = 600 ;
offset2 = 1700 ;

d = a ( I_row - offset1 : I_row+offset1, I_col-offset2 : I_col+offset2 ) ;
imresize(a,0.2) ; 

nocols = sum (avg >= 17) ;                                                 % Looking for the number of columns with a pixel value greater than 17 (i.e. part of the beam and not the background)

surf(a, 'Edgecolor' , 'none' )
