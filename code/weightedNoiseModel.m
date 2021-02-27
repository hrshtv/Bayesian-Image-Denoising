%{
Description: Implementation of the weighted noise models (Complex Gaussian and Rician)
Inputs: 
    img_noisy: The original noisy image
    img_current: The image being denoised
    model_name: "complex_gaussian" or "rician"
    sigma: std deviation
    alpha: The weighting term
Outputs:
    weighted_term_noise: [N, N, 2] array containing the stacked likelihood and it's gradient
%}

function weighted_term_noise = weightedNoiseModel(img_noisy, img_current, model_name, sigma, alpha)
    
    if model_name == "complex_gaussian"
        noise_val  = (1/sigma^2)*(abs(img_noisy - img_current).^2); % This term depends only on the noise model, and not the prior model
        noise_grad = (2/sigma^2)*(img_current - img_noisy); % Gradient of the noise term 2*(x_i - y_i) from the slides
        weighted_term_noise = (1-alpha)*cat(3, noise_val, noise_grad); % [256, 256, 2] stacked along 3rd dimension for vectorization
    
    elseif model_name == "rician"
        
        % Rename according to slide convention:
        y = img_noisy;
        x = img_current;
        weight = 1-alpha;
        
        yx = y .* x; % i^th element is y_i*x_i
        I0 = besseli(0, weight*(1/sigma^2)*yx);
        I1 = besseli(1, weight*(1/sigma^2)*yx);
        
        noise_val  = weight*(0.5/sigma^2)*(y.^2 + x.^2) - log(I0); 
        noise_grad = weight*(1/sigma^2)*(x - ((I1 ./ I0).*y));
        weighted_term_noise = cat(3, noise_val, noise_grad); % [256, 256, 2] stacked along 3rd dimension for vectorization
    
    end
end

