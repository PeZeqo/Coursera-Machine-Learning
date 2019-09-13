function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

cost_mult = alpha / m

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    X_temp = X * theta;

    temp_vec = (X_temp - y);

    % temp_sum = sum(temp_vec);

    % temp_sum = temp_sum * cost_mult;
    
    temp_theta = zeros(size(theta))

    for i = 1:size(theta),
        temp_sum = transpose(temp_vec) * X(:,i);
        
        temp_theta(i) = temp_sum * cost_mult
    end

    
    theta = theta - temp_theta

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

plot(J_history)

end
