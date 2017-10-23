data=load('C:\Users\akshaybahadur21\Desktop\Poker_Hand\poker-hand-testing.data');
X=data(:,1:10);
y=data(:,11);

%No of output categories
input_layer_size  = 10;  
hidden_layer_size = 25;   
num_labels = 10; 

m = size(X, 1);

%theta1 should be 18*86
%theta2 should be 10*19

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);