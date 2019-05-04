function [U] =hpr(A,B)
%take an arry (A) and removes all the hot pixels (B)


    C = find(A>=B)  ;                                                      % This finds the elements in A that are greater than or equal to B
    A2 = conv2( A , [1 1 1 ; 1 0 1 ; 1 1 1]/8 , 'same' ) ;                 % This creates a new image where every pixel is the average of its 8 neighbours in A
    A(C) = A2(C) ;                                                         % This assigns all the elements in A that are larger where found using C to be their equivalent values in A2
    
    U= A ;                                                                 % Sets the output to be equal to the new value for A
end


