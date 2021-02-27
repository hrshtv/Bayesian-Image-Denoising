%{
Description: Returns the weighted negative log posterior (our objective function) and the gradients of the objective function for each pixel
Inputs:
    img_noisy  : "y_i" from the slides, the noisy image
    img_current: "x_i" from the slides, this is our current denoised image, initially it will be same as img_noisy
    alpha: Our weighting factor, lies in [0,1]
    prior_potential: Specifies which energy/potential function to use in the prior. 
                     Choices: "quadratic", "huber"
    noise_model_name: "complex_gaussian" or "rician"
    gamma: Specify when using Huber or DAF penalty, lies in (0, Inf)
Outputs:
    obj_fn: Weighted negative log posterior
    grads: A NxN array containing the gradients for each pixel
%}

function [obj_fn, grads] = objectiveAndGrads(img_noisy, img_current, alpha, prior_potential, noise_model_name, sigma, gamma)
    
    % We use a circularly symmetric univariate Gaussian noise model:
    % sigma = 1; % Since we will tune alpha later, sigma can be anything
    
    % Likelihood:
    weighted_term_noise = weightedNoiseModel(img_noisy, img_current, noise_model_name, sigma, alpha); % [256, 256, 2] stacked along 3rd dimension for vectorization
    
    % All potential functions are functions of these diffs:
    [diff_top, diff_bot, diff_left, diff_right] = fourNeighborSystem(img_current);
    
    % Quadratic Prior:
    if prior_potential == "quadratic"
        term_prior = quadraticPrior(diff_top) + quadraticPrior(diff_bot) + quadraticPrior(diff_left) + quadraticPrior(diff_right);
    % Huber Prior:
    elseif prior_potential == "huber"
        term_prior = huberPrior(diff_top, gamma) + huberPrior(diff_bot, gamma) + huberPrior(diff_left, gamma) + huberPrior(diff_right, gamma);
    % Discontinuity Adaptive Function Prior:
    elseif prior_potential == "daf"
        term_prior = dafPrior(diff_top, gamma) + dafPrior(diff_bot, gamma) + dafPrior(diff_left, gamma) + dafPrior(diff_right, gamma);
    end
    % term_prior contains the penalty values as well as the gradients stacked along the third dimension 
    
    % Weighted Negative Log Posterior and its gradients: Common computation for all priors
    wnlp_and_grads = weighted_term_noise + alpha*term_prior; 
    wnlp  = wnlp_and_grads(:, :, 1);
    grads = wnlp_and_grads(:, :, 2);
     
    obj_fn = sum(wnlp(:)); % Sum since actual posterior is a product, thus summing up the log terms 
   
end

