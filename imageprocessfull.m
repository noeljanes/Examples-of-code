clear

g = 31 ;                                                                   % Declaring the number of images in the folder

%cols = zeros(g) ;                                                         % Declares a zero array of teh required length

% Distances travelled by the flags

%dist1 = [0 , -0.3556 , -0.7112 , -1.143, -1.4946, -1.8462, -2.1978, -2.5494, -2.901, -3.2526, -3.6042, -3.9558, -4.3074, -4.663, -5.0186, -5.3742 , 0 , 0.3556 , 0.6858, 1.0414, 1.397, 1.7526, 2.1082, 2.4638, 2.8194, 3.175, 3.5306, 3.8862, 4.2418, 4.5974, 4.953] ;

dist1 = [0 , -0.3556 , -0.7112 , -1.143, -1.4946, -1.8462, -2.1978, -2.5494, -2.901, -3.2526, 0 , 0.3556 , 0.6858, 1.0414, 1.397, 1.7526, 2.1082, 2.4638, 2.8194, 3.175] ;

%dist2 = [0 , -0.3556, -0.7112, -1.0668, -1.4224, -1.778, -2.1336, -2.4892, -2.8448, -3.2004, -3.556, -3.9116, -4.2672, -4.6226, -4.9784, 0 , 0.3556, 0.7112, 1.0668, 1.4224, 1.778, 2.1336, 2.4892, 2.8448, 3.2004, 3.556, 3.9116, 4.2672, 4.6226, 4.9784] ; 
dist2 = [0 , -0.3556, -0.7112, -1.0668, -1.4224, -1.778, -2.1336, -2.4892, -2.8448, -3.2004, 0 , 0.3556, 0.7112, 1.0668, 1.4224, 1.778, 2.1336, 2.4892, 2.8448, 3.2004] ; 

dist3 = [ 0 , -0.3556, -0.7112, -1.0668, -1.4224, -1.8034, -2.159 , -2.5146 , -2.8448 , -3.2004, 0 , 0.3556, 0.7112, 1.0668, 1.4478, 1.778, 2.1336 , 2.4892 , 2.8448 , 3.2004] ;


% Counters
counter1 = 0 ;
counter2 = 0 ;
counter3 = 0 ;


%% Start of the section for the unpolished flag

k = rgb2gray(imread('IMG_9209.cr2')) ;                                     % Importing and greyscaling the image
k = imrotate(k,10) ;                                                       % Rotating the image to align it correctly, with the center beam being horizontal
[M , I] = max(k(:)) ;     
[I_row2 , I_col2] = ind2sub(size(k),I) ; 

for A = 10:19
   
    counter1 = counter1+1 ;
    [nocols] = imageproc(A, I_row2,I_col2)     ;                           % Inputs into the corresponding function
    cols1(counter1) = nocols ; 
    
end

% A second for loop is used to avoid the identical images at the end of one
% direction of travel

for A = 26:35
   
    counter1 = counter1+1 ;
    [nocols] = imageproc(A, I_row2,I_col2)     ;                           % Inputs into the corresponding function
    cols1(counter1) = nocols ; 
    
end

%% Start of the section for the polished flag

k = rgb2gray(imread('IMG_9256.cr2')) ;                                     % Importing and greyscaling the image
k = imrotate(k,10) ;                                                       % Rotating the image to align it correctly, with the center beam being horizontal
[M , I] = max(k(:)) ;     
[I_row3 , I_col3] = ind2sub(size(k),I) ; 

for A = 42:51
    
    counter2 = counter2+1 ;
    [nocols] = imageproc(A, I_row3,I_col3)     ;                           % Inputs into the imageprocessing function
    cols2(counter2) = nocols ; 
    
end

% A second for loop is used to avoid the identical images at the end of one
% direction of travel
for A = 57:66
    
    counter2 = counter2+1 ;
    [nocols] = imageproc(A, I_row3,I_col3)     ;                           % Inputs into the imageprocessing function
    cols2(counter2) = nocols ; 
    
end


k = rgb2gray(imread('IMG_9281.cr2')) ;                                     %Importing and greyscaling the image
k = imrotate(k,10) ;                                                       % Rotating the image to align it correctly, with the center beam being horizontal
[M , I] = max(k(:)) ;     
[I_row4 , I_col4] = ind2sub(size(k),I) ; 


%% Processing the images for the razorblade flag
for A = 72:91
    
    counter3 = counter3+1 ;
    [nocols] = imageproc(A, I_row4,I_col4)     ;                           % Inputs into the imageprcessing function
    cols3(counter3) = nocols ; 
    
end

%% Plotting a scatter plot of the number of columns with a pixel value greater than 17

scatter(dist1, cols1, 40, 'o','r') ;

hold on
l1=polyval(polyfit(dist1,cols1,1),dist1) ;                                 % Makes a trendline for this flags travel data

scatter(dist2,cols2, 40, '.','b') ;
l2=polyval(polyfit(dist2,cols2,1),dist2) ;                                 % Makes a trendline for this flags travel data

scatter(dist3,cols3, 40, 'x','m') ;
l3=polyval(polyfit(dist3,cols3,1),dist3) ;                                 % Makes a trendline for this flags travel data


plot(dist1,l1,'-r',  dist2,l2,':b', dist3,l3,'--m') ;                      % Plotting the trendlines on the scatter plot

xlabel('Relative position of each flag (mm)') ;
ylabel('Number of columns with an average pixel value greater than 17') ;
legend({'Unpolished cylindrical copper flag', 'Polished cylindrical copper flag', 'Razor blade','Trendline for the Unpolished cylindrical copper flag','Trendline for the Polished cylindrical copper flag','Trendline for the Razor Blade flag'},'Location','southwest')
grid on
grid minor
hold off

%% Calculating the chi square values for each plot
% Declaring initial chi-squared variables

chi1 = 0 ;                                                                 
chi2 = 0 ;
chi3 = 0 ;

for b=1:20
    
    chi1 = chi1 + ((cols1(b)-l1(b))^(2))/l1(b) ;
    chi2 = chi2 + ((cols2(b)-l2(b))^(2))/l2(b) ;
    chi3 = chi3 + ((cols3(b)-l3(b))^(2))/l3(b) ;
    
end