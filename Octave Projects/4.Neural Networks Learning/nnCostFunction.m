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

%compute hypothesis 
X = [ones(m, 1) X];
temp1=sigmoid(X*Theta1');
temp2=[ones(size(temp1,1), 1) temp1];
H=sigmoid(temp2*Theta2');

%reshape y
Y=zeros(m,num_labels);
temp3=[1:num_labels];
for i=1:m
Y(i,:)=(temp3==y(i));
end
        
% You need to return the following variables correctly 
J = 0;
for i=1:m
 for k=1:num_labels
  J=J+(  -(Y(i,k)*log(H(i,k)))-(1-Y(i,k))*log(1-H(i,k))  );
  end
end

%unregularized cost function
%J=(1/m)*J;

%regularized cost function
Theta1f=Theta1(:, 2:end);
Theta2f=Theta2(:, 2:end);
J=(1/m)*J+(lambda/(2*m))*( sum(sum(Theta1f.^2))+sum(sum(Theta2f.^2)) )  ; 



% compute gradient of the neural network
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Delta_1=zeros(size(Theta1));
Delta_2=zeros(size(Theta2));

for t=1:m
%step1
a_1=X(t,:)';
z_2=Theta1*a_1;
a_2=sigmoid(z_2);
a_2=[1;a_2];
z_3=Theta2*a_2;
a_3=sigmoid(z_3);

%step2
delta_3=a_3-Y(t,:)';

%step3
delta_2=(Theta2'*delta_3).*[0;sigmoidGradient(z_2)];

%step4
delta_2=delta_2(2:end);
Delta_1=Delta_1+delta_2*a_1';
Delta_2=Delta_2+delta_3*a_2';
end

%gradient of the unregularized neural network
Theta1_grad=(1/m)*Delta_1;
Theta2_grad=(1/m)*Delta_2;

%gradient of the regularized neural network
Theta1_grad(:, 2:end)=Theta1_grad(:, 2:end)+(lambda/m)*Theta1(:, 2:end);
Theta2_grad(:, 2:end)=Theta2_grad(:, 2:end)+(lambda/m)*Theta2(:, 2:end);




% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

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



















% -------------------------------------------------------------

% =========================================================================




end
