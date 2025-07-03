import Data_Handling
import matrix_math
import math

# Initialize weights and bias
def initialize_weights_and_bias(num_features, num_hidden):
    weights = [[0.01 for _ in range(num_hidden)] for _ in range(num_features)]
    biases = [0.01 for _ in range(num_hidden)]
    return weights, biases

# Forward pass functions
def ReLu(x):
    if x < 0:
        return 0
    return x

def Sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward_pass(segment, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_raw = matrix_math.matrix_mult(segment, weights_input_hidden)
    hidden_raw = matrix_math.add_matrix(hidden_raw, bias_hidden)
    hidden_activated = ReLu(hidden_raw)

    output_raw = matrix_math.dot_product(hidden_activated, weights_hidden_output) + bias_output
    output = Sigmoid(output_raw)

    return output

# Backpropagation functions
def ReLu_derivative(x):
    return 1 if x > 0 else 0

def sigmoid_derivative(x):
    return x * (1 - x)
    
epsilon = 1e-15

# Loss function
def loss_function(output, true):
    total_loss = 0
    for y, t in zip(output, true):
        clamped_output = max(min(y, 1 - epsilon), epsilon)  # Clamp output to avoid log(0)
        loss += - (t * math.log(clamped_output) + (1 - t) * math.log(1 - clamped_output))
        total_loss += loss
    return total_loss / len(output)  # Average loss over all outputs

def find_gradient(output, true):
    output_gradient = []
    for y, t in zip(output, true):
        clamped_output = max(min(y, 1 - epsilon), epsilon)
        output_gradient = ((-t/clamped_output) + (1-clamped_output)/(1-t)) * sigmoid_derivative(clamped_output)
        output_gradient.append(output_gradient)
    return output_gradient

def initialize_gradient_matrix_like(matrix):
    return [[0 for i in range(len(matrix[0]))] for i in range(len(matrix))]

def initialize_gradient_vector_like(vector):
    return [0 for _ in range(len(vector))]

def backprop(segments, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, true_labels):
    grad_w_input_hidden_sum = initialize_gradient_matrix_like(weights_input_hidden)
    grad_b_hidden_sum = initialize_gradient_vector_like(bias_hidden)
    grad_w_hidden_output_sum = initialize_gradient_matrix_like(weights_hidden_output)
    grad_b_output_sum = initialize_gradient_vector_like(bias_output)

    for segment, true_label in zip(segments, true_labels):
        output, hidden_activations = forward_pass(segment, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
        grad_output_layer = find_gradient([output], [true_label])[0]

        # Calculate gradients for output layer
        grad_w_hidden_output = matrix_math.matrix_mult(matrix_math.transpose(hidden_activations), grad_output_layer)
        grad_b_output = grad_output_layer

        # Calculate error for hidden layer
        hidden_error = matrix_math.elementwise_mult(matrix_math.matrix_mult(grad_output_layer, matrix_math.transpose(weights_hidden_output)), ReLu_derivative(hidden_activations))

        # Calculate gradients for input-hidden layer
        grad_w_input_hidden = matrix_math.matrix_mult(matrix_math.transpose(segment), hidden_error)
        grad_b_hidden = hidden_error

        grad_w_input_hidden_sum += grad_w_input_hidden
        grad_b_hidden_sum += grad_b_hidden
        grad_w_hidden_output_sum += grad_w_hidden_output
        grad_b_output_sum += grad_b_output

    n = len(segments)
    return (
        grad_w_input_hidden_sum / n,
        grad_b_hidden_sum / n,
        grad_w_hidden_output_sum / n,
        grad_b_output_sum / n
    )
