function [nocols] = imageproc(A,I_row,I_col)
    % Function inputs the number and pulls that file from the folder. 
    % This function initially imports and greyscales the image and then
    % cuts it down to the desired shape and size
    
    name = sprintf('IMG_92%.0f.cr2', A)

    a = rgb2gray(imread(name)) ;                                           %Importing and greyscaling the image

    a = imrotate(a,10) ;                                                   % Rotating the image to align it correctly, with the center beam being horizontal
    
    offset1 = 700 ;
    offset2 = 1700 ;

    d = a ( I_row - offset1 : I_row+offset1, I_col-offset2 : I_col+offset2 ) ;
    avg = mean(d) ;
    nocols = sum (avg >= 17) ;
end

