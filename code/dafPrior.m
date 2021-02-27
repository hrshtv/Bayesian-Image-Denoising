%{
Description: This function computes the Discontinuity Adaptive Function and it's gradient
Inputs: 
    x: The difference between the pixel and it's neighbours. Can be present given as an N*N matrix of differences for vectorization
    gamma: A parameter, lies in (0, Inf)
Outputs:
    prior_info: The penalty values and the respective gradients, stacked along the thrd dimension as a [N, N, 2] array
%}

function prior_info = dafPrior(x, gamma)
    g  = gamma*abs(x) - (gamma^2)*log(1 + (1/gamma)*abs(x));
    dg = (sign(x).*(gamma*abs(x))).*(1./(gamma + abs(x)));
    prior_info = cat(3, g, dg);
end

