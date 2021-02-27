%{
Description: This function computes the Huber penalty/potential and it's gradient
Inputs: 
    x: The difference between the pixel and it's neighbours. Can be present given as an N*N matrix of differences for vectorization
    gamma: A parameter, lies in (0, Inf)
Outputs:
    prior_info: The penalty values and the respective gradients, stacked along the thrd dimension as a [N, N, 2] array
%}

function prior_info = huberPrior(x, gamma)
    abs_x = abs(x);
    mask_1 = abs_x <= gamma;
    mask_2 = abs_x > gamma;
    
    g = zeros(size(x));
    g(mask_1) = 0.5*(abs_x(mask_1).^2); % abs_x used to support complex valued images 
    g(mask_2) = gamma*abs_x(mask_2) - 0.5*(gamma^2);
    
    dg = zeros(size(x));
    dg(mask_1) = x(mask_1);
    dg(mask_2) = gamma*sign(x(mask_2));
    
    prior_info = cat(3, g, dg);
end

