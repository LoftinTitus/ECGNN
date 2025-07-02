import Data_Handling
import math

# Initialize weights and bias
def initialize_weights_and_bias(all_data):
    weights = [0.01] * all_data  # Initialize weights to small values
    bias = 0.01  # Initialize bias to a small value
    return weights, bias

# Forward pass functions
def ReLu(x):
    if x < 0:
        return 0
    return x

def Sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward_pass_v2(all_data, weights, bias):
    output = []
    for row in all_data:
        # Extract 2nd and 3rd columns (indices 1 and 2)
        inputs = [row[1], row[2]] if len(row) > 2 else []
        
        neuron_output = bias  # Start with bias
        for j in range(min(len(inputs), len(weights))):
            neuron_output += inputs[j] * weights[j]
        output.append(Sigmoid(neuron_output))
    return output

# Loss function
def loss_function(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    
    total_loss = 0.0
    for i in range(len(predictions)):
        error = targets[i] - predictions[i]
        total_loss += error ** 2  # Mean Squared Error
    return total_loss / len(predictions)  # Return average loss

# Backpropagation functions
def ReLu_derivative(x):
    return 1 if x > 0 else 0

def sigmoid_derivative(x):
    return x * (1 - x)

def backprop(all_data, weights, output, bias, learning_rate=0.01):
    for row in all_data:
        inputs = [row[1], row[2]] if len(row) > 2 else []
        
        # Forward pass  
        neuron_output = bias
        for j in range(min(len(inputs), len(weights))):
            neuron_output += inputs[j] * weights[j]
        output = Sigmoid(neuron_output)
        
        # Calculate error
        target = row[0]  # Assuming the first column is the target value
        error = target - output
        
        # Backward pass
        d_output = error * sigmoid_derivative(output)
        
        # Update weights and bias
        for j in range(min(len(inputs), len(weights))):
            weights[j] += learning_rate * d_output * inputs[j]
        bias += learning_rate * d_output
    return weights, bias

def update_weights_and_bias(weights, bias, learning_rate=0.01):
    for i in range(len(weights)):
        weights[i] += learning_rate * weights[i]  # Update weights
    bias += learning_rate * bias  # Update bias
    return weights, bias

def train_model(all_data, epochs=100, learning_rate=0.01):
    weights, bias = initialize_weights_and_bias(all_data)
    
    for epoch in range(epochs):
        output = forward_pass_v2(all_data, weights, bias)
        loss = loss_function(output, [row[0] for row in all_data])  # Assuming first column is target
        
        # Backpropagation
        weights, bias = backprop(all_data, weights, output, bias, learning_rate)
        
        if epoch % 10 == 0:  # Print loss every 10 epochs
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return weights, bias