%{
Description: This function computes the RRMSE between an image and a reference image
Inputs: 
    ref: Ground truth/Reference image
    img: The image to be compared
Outputs:
    error: The Relative Root Mean Squared Error
%}
  
function error = rrmse(ref, img)

    diff = abs(ref) - abs(img);
    diff_sq = diff.^2;
    ref_sq = abs(ref).^2; % abs used for supporting complex valued data
    
    % Sum over all the pixels
    numerator = sum(diff_sq(:));
    denominator = sum(ref_sq(:));
    
    error = sqrt(numerator/denominator);

end

