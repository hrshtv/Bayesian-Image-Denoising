%{
Description: This function computes the quadratic penalty/potential and it's gradient
Inputs: 
    x: The difference between the pixel and it's neighbours. Can be present given as an N*N matrix of differences for vectorization
Outputs:
    prior_info: The penalty values and the respective gradients, stacked along the thrd dimension as a [N, N, 2] array
%}

function prior_info = quadraticPrior(x)
    g  = abs(x).^2;  % Value of the penalty, abs() used to support complex data
    dg = 2*x; % Gradient of the penalty
    prior_info = cat(3, g, dg); % Concatenate along the third dimension
end
