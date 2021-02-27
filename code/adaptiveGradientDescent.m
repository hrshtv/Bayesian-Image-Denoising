%{ 
Description: Gradient Descent with Dynamic Step Size
Inputs:
    img_noisy: The image to be denoised
    img_initial: The inital solution for the denoised image, in most cases it's the noisy image itself
    prior_potential: "quadratic"
    noise_model_name: "complex_gaussian" or "rician"
    alpha: The weighting factor, lies in [0,1]
    gamma: Parameter for the Huber and the DAF penalty, lies in (0, Inf)
    n_iters: Number of iterations (An upper bound on the actual number of updates)
    lr: The learning rate/step size, to be updated dynamically based on the objective function values
Outputs:
    img_denoised: The image after all iterations have been completed
    obj_values: For keeping track of the objective function throughout the training
%}

function [img_denoised, obj_values] = adaptiveGradientDescent(img_noisy, img_initial, prior_potential, noise_model_name, sigma, alpha, gamma, n_iters, lr)
    
    lr_stop = 1e-10; % If the lr decreases beyond this, stop the loop (else infinite loop is possible)
    
    % Initialization:
    obj_values = zeros([n_iters, 1]); % Pre-allocated for speed
    obj_fn = Inf;
    img_current = img_initial;
    
    % Gradient Descent:
    for i = 1:n_iters
        
        % Store the previous value of the objective function
        obj_prev = obj_fn;
        
        % Candidate for update:
        [obj_new, grads] = objectiveAndGrads(img_noisy, img_current, alpha, prior_potential, noise_model_name, sigma, gamma);
        
        if obj_new >= obj_prev % Worse/same as previous solution
            % Decrease the learning rate and make no updates
            if (lr <= lr_stop)
                fprintf("Feedback: Stopping at %ith iteration because step-size/learning-rate became too low\n", i);
                obj_values = obj_values(1:i-1);
                break;
            end
            lr = 0.5*lr; % Decrease LR by 50%, don't update
        else % Improved
            % Make the actual updates:
            img_current = img_current - lr*grads;
            obj_fn = obj_new;
            lr = 1.1*lr; % Increase LR by 10%
        end
        obj_values(i) = obj_fn; % Don't update with obj_new
    end
    
    img_denoised = img_current;
    
end
