import Data_Handling
import math


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

# Backpropagation functions
def ReLu_derivative(x):
    return 1 if x > 0 else 0

def sigmoid_derivative(output):
    return output * (1 - output)