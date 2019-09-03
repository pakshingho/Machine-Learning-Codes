function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Part 1:
% 1 - Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5).
% This is most easily done using an eye() matrix of size num_labels, with vectorized indexing by 'y'.
% A useful variable name would be "y_matrix", as this...

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:); % y_matrix is (m x c), where 'c' is the number of labels

% 2 - Perform the forward propagation:

a1 = [ones(m, 1) X];            % a1 is (m x n), where 'n' is the number of features including the bias unit
z2 = a1 * Theta1';              % Theta1 is (h x n) where 'h' is the number of hidden units
a2 = [ones(m, 1) sigmoid(z2)];  % a2 is (m x (h + 1))
z3 = a2 * Theta2';              % Theta2 is (c x (h + 1)), where 'c' is the number of labels
a3 = sigmoid(z3);               % a3 is (m x c)
% keyboard
% The values can be inspected by adding the "keyboard" command within your 
% for-loop. This exits the code to the debugger, where you can inspect the 
% values. Use the "return" command to resume execution.
%[~, p] = max(a3, [], 2);        % p is a vector of size (m x 1)

% 3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
% using a3, your y_matrix, and m (the number of training examples). 
% Note that the 'h' argument inside the log() function is exactly a3. 
% Cost should be a scalar value. Since y_matrix and a3 are both matrices, 
% you need to compute the double-sum.

J = 1/m * sum(sum(-y_matrix .* log(a3) - ( 1 - y_matrix ) .* log( 1 - a3 )));

% 4 - Compute the regularized component of the cost according to ex4.pdf Page 6,
% using ?1 and ?2 (excluding the Theta columns for the bias units), along with ?,
% and m. The easiest method to do this is to compute the regularization terms separately,
% then add them to the unregularized cost from Step 3.

% method 1 (not correct if later need to call Theta1 & Teta2 again):
%Theta1(:,1) = 0;
%Theta2(:,1) = 0;
%reg1 = sum(sum(Theta1.^2));
%reg2 = sum(sum(Theta2.^2));
% method 2:
reg1 = sum(sum(Theta1(:,2:end).^2));
reg2 = sum(sum(Theta2(:,2:end).^2));
J = J + lambda/(2*m)*(reg1 + reg2);

% Part 2:
% Let:
% m = the number of training examples
% n = the number of training features, including the initial bias unit.
% h = the number of units in the hidden layer - NOT including the bias unit
% r = the number of output classifications
d3 = a3 - y_matrix; % (m x r)
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2); % The size is (m x r)?(r x h) --> (m x h). The size is the same as z2, as must be.
Delta1 = d2' * a1;                                % The size is (h x m)?(m x n) --> (h x n)
Delta2 = d3' * a2;                                % The size is (r x m)?(m x [h+1]) --> (r x [h+1])
Theta1_grad = 1/m * Delta1;                       % same size as their respective Deltas, just scaled by 1/m.
Theta2_grad = 1/m * Delta2;
temp_Theta1 = Theta1; 
temp_Theta2 = Theta2;
temp_Theta1(:,1)=0;
temp_Theta2(:,1)=0;
Theta1_grad = Theta1_grad + lambda/m * temp_Theta1;
Theta2_grad = Theta2_grad + lambda/m * temp_Theta2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
