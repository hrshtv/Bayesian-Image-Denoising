%{
Description: 
    4-Neighbor system with wrap-around at the boundaries. 
    We compute neighbours of all pixels together for vectorization.
Inputs:
    img: A NxN array
Outputs:
    The absolute differences (as NxN arrays) of each pixel from it's neighbors in  each of the 4 directions 
%}
  
function [diff_top, diff_bot, diff_left, diff_right] = fourNeighborSystem(img)
    n_top   = circshift(img, -1, 1); % Shift circularly by 1 pixel downwards
    n_bot   = circshift(img,  1, 1); % Shift circularly by 1 pixel upwards
    n_left  = circshift(img,  1, 2); % Shift circularly by 1 pixel to the right
    n_right = circshift(img, -1, 2); % Shift circularly by 1 pixel to the left
    
    diff_top   = img - n_top; % Deviation, "(x - y_i)" from the slides
    diff_bot   = img - n_bot;
    diff_left  = img - n_left;
    diff_right = img - n_right;
end

