clear
%% 1 Input data and setup

    rec=VideoReader('MVI_5079.MOV') ;                                      % Reads in the recording of interest
    %Dark=VideoReader('darkcurrent.avi');                                  % Reads in the Dark current frames 

    %cutoff=250    ;                                                       % How many frames we want the final to be created up from 

    i = struct('cdata','single') ;                                         % A struct to save each frame after image correction 
    nframes=int16(rec.FrameRate*rec.Duration) ;                            % Works out the number of frames in the full video

    max_col= zeros( nframes , 4);
    max_row= zeros( nframes, 4);                                           % Creates a vectors of length nframes with zeros for storage, and with 3 rows to store the index of each one aswell
    max_val = zeros( nframes , 3 ) ;      

    %darkframe=darkfunction(Dark,rec);  

    qualitycols=zeros( nframes,5);
    qualityrows=zeros( nframes,5);

    sum_img = zeros( 1049 , rec.Width ) ;

%% 2 Picks out and plots each frame and cleans each frame
for k = 1 : nframes
   
    this_frame = readFrame(rec) ;                                          % Reads in each frame
    this_frame = single(rgb2gray(this_frame)) ;                            % Changes this_frame from a uint8 array to a double
    [ a , b ] = size(this_frame) ;
    this_frame(1050:a, : ) = [] ;
    clear a ;
    clear b ;
    
    
    % 2.1 Looking for the hotpixels
    %{
    figure()
    histogram(this_frame, 75)
    set( gca , 'Yscale' , 'log' );                                         % Changes the y-scale to be logarithmic 
    set(gca, 'Xscale' );                                                   % Changes the x axis to br logarithmic
    xlabel( 'Pixel Value' ) ;
    ylabel( ' Log of the pixel count ' ) ;
    %}
        
    % 2.2 Cleaning the image frame by removing hot pixels and the
    % darkframe
    %tfdf = this_frame - darkframe ;                                       % Removes the mean pixel value from the frame
    [ tfclean ] = hpr ( this_frame , 150 );                                % Calls the hot pixel removal function, 150 was chosen as a cutoff value as it was the highest pixel value that removed all the hot pixels from the image   
    
    % 2.4 Finding the peak brightness in the frame as well as its position
    % index
        
    [ Max , I ] = max(this_frame(:)) ;
	[I_row, I_col] = ind2sub(size(this_frame),I) ;
    max_val( k,: ) = [ Max  I_row  I_col ] ;

    
    % 2.5 Finding the peak column position 
    
    [ xce , pce , xco , pco , e , o ] = tidycols(tfclean) ;                % Finds the highest sum column in the cleaned image
    
    [ le , we , lo , wo ] = peaks ( e , o ) ;
    qualitycols(k,:) = [ le , we , lo , wo , k ] ;
    
    i(k).cdata = tfclean ;                                                 % Saves each image to a structure
        
    % 2.6 Plotting the sum of each column to get fwhm
                 
    %{
    figure()
    subplot(3,1,1)
    plot( e , '.' )
    set( gca , 'Visible' , 'on' )
    subplot(3,1,2)
    plot( o , '.' )
    set( gca , 'Visible' , 'on' )
    subplot(3,1,3)
    plot(Sumcol, '.')
    set( gca , 'Visible' , 'on' )
    %}
            
    % 2.7 Creating the summed image for the run
    sum_img = sum_img + tfclean ;                                          % Adds this frame to the overall summed image
end

%% 3 Creating the average image for the run

    avg_img = sum_img ./ single(nframes) ;                                 % Averages the summed image for all the frames
    
    %{
    figure();
    imagesc( avg_img )
    colorbar ;
    caxis() ;
    colormap(gray) ;
    title('avg_img');
    %}
    
%% 4 Image selection
    
    imagerank_even = sortrows ( qualitycols , 2 ) ;                        % Sorts by width of even cols
    imagerank_odd = sortrows ( qualitycols , 4 ) ;                         % Sorts by width of odd cols
    imagerank_even ( : , [ 3 , 4 ] ) = [] ;                                % Removes odd columns
    imagerank_odd ( : , [ 1 , 2 ] ) = [] ;                                 % Removes even columns
    imagerank_even (( cutoff+1:nframes ), : ) = [] ;                       % Creates an array with the top cutoff frames for even
    imagerank_odd (( cutoff+1:nframes ), : ) = [] ;                        % Creates an array with the top cutoff frames for odd
    
   % [f] = finalimage ( imagerank_even , i, rec) ;
    [final] = shift ( i , imagerank_odd ) ;     
             