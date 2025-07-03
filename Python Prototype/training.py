import math
import Brains

def update_weights_and_bias(weights, bias, learning_rate=0.01):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            weights[i][j] -= learning_rate * weights[i][j]
    for i in range(len(bias)):
        bias[i] -= learning_rate * bias[i]
    return weights, bias

def train_model(all_data, epochs=100, learning_rate=0.01):
    num_features = len(all_data[0]) - 1  # Assuming last element is the label
    num_hidden = 2  # hidden layer size

    weights_input_hidden, bias_hidden = Brains.initialize_weights_and_bias(num_features, num_hidden)
    weights_hidden_output, bias_output = Brains.initialize_weights_and_bias(num_hidden, 1)  # Binary classification

    for epoch in range(epochs):
        for segment in all_data:
            inputs = segment[:-1]
            true_label = segment[-1]

            output, hidden_activations = Brains.forward_pass(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

            grad_w_input_hidden, grad_b_hidden, grad_w_hidden_output, grad_b_output = Brains.backprop(
                inputs, hidden_activations, output, true_label, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
            )

            weights_input_hidden, bias_hidden = update_weights_and_bias(weights_input_hidden, bias_hidden, grad_w_input_hidden, grad_b_hidden, learning_rate)
            weights_hidden_output, bias_output = update_weights_and_bias(weights_hidden_output, bias_output, grad_w_hidden_output, grad_b_output, learning_rate)

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

def evaluate_model(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, test_data):
    correct_predictions = 0
    for segment in test_data:
        inputs = segment[:-1]
        true_label = segment[-1]

        output, _ = Brains.forward_pass(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
        predicted_label = 1 if output >= 0.5 else 0

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    return accuracy