use crate::matrix_math::*;

/// Small constant to prevent division by zero and log(0) in loss calculations
const EPSILON: f64 = 1e-15;

/// Initialize weights and bias for the neural network
/// 
/// # Arguments
/// * `num_features` - Number of input features
/// * `num_hidden` - Number of hidden neurons
/// 
/// # Returns
/// * Tuple containing (weights_matrix, bias_vector)
pub fn initialize_weights_and_bias(num_features: usize, num_hidden: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let weights = vec![vec![0.01; num_hidden]; num_features];
    let biases = vec![0.01; num_hidden];
    (weights, biases)
}

/// ReLU activation function
/// Returns 0 for negative inputs, x for positive inputs
pub fn relu(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

/// Sigmoid activation function
/// Maps any real number to a value between 0 and 1
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Apply ReLU activation element-wise to a vector
pub fn apply_relu(input: &Vec<f64>) -> Vec<f64> {
    input.iter().map(|&x| relu(x)).collect()
}

// Forward pass function
pub fn forward_pass(
    segment: &Vec<Vec<f64>>,
    weights_input_hidden: &Vec<Vec<f64>>,
    bias_hidden: &Vec<f64>,
    weights_hidden_output: &Vec<f64>,
    bias_output: f64,
) -> (f64, Vec<f64>) {
    // Calculate hidden layer raw values
    let hidden_raw_matrix = matrix_multiply(segment, weights_input_hidden);
    let hidden_raw = &hidden_raw_matrix[0]; // Assuming single sample
    
    // Add bias to hidden layer
    let hidden_with_bias: Vec<f64> = hidden_raw.iter().zip(bias_hidden.iter())
        .map(|(h, b)| h + b)
        .collect();
    
    // Apply ReLU activation
    let hidden_activated = apply_relu(&hidden_with_bias);
    
    // Calculate output
    let output_raw = dot_product(&hidden_activated, weights_hidden_output) + bias_output;
    let output = sigmoid(output_raw);
    
    (output, hidden_activated)
}

// Derivative functions
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

// Loss function (Binary Cross-Entropy)
pub fn loss_function(output: &Vec<f64>, true_labels: &Vec<f64>) -> f64 {
    let mut total_loss = 0.0;
    
    for (y, t) in output.iter().zip(true_labels.iter()) {
        let clamped_output = y.max(EPSILON).min(1.0 - EPSILON);
        let loss = -(t * clamped_output.ln() + (1.0 - t) * (1.0 - clamped_output).ln());
        total_loss += loss;
    }
    
    total_loss / output.len() as f64
}

// Find gradient for output layer
pub fn find_gradient(output: &Vec<f64>, true_labels: &Vec<f64>) -> Vec<f64> {
    let mut output_gradient = Vec::new();
    
    for (y, t) in output.iter().zip(true_labels.iter()) {
        let clamped_output = y.max(EPSILON).min(1.0 - EPSILON);
        let grad = ((-t / clamped_output) + (1.0 - t) / (1.0 - clamped_output)) * sigmoid_derivative(clamped_output);
        output_gradient.push(grad);
    }
    
    output_gradient
}

// Apply ReLU derivative element-wise
pub fn apply_relu_derivative(input: &Vec<f64>) -> Vec<f64> {
    input.iter().map(|&x| relu_derivative(x)).collect()
}

// Backpropagation function
pub fn backprop(
    segments: &Vec<Vec<Vec<f64>>>,
    weights_input_hidden: &Vec<Vec<f64>>,
    bias_hidden: &Vec<f64>,
    weights_hidden_output: &Vec<f64>,
    bias_output: f64,
    true_labels: &Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, f64) {
    let mut grad_w_input_hidden_sum = initialize_gradient_matrix_like(weights_input_hidden);
    let mut grad_b_hidden_sum = initialize_gradient_vector_like(bias_hidden);
    let mut grad_w_hidden_output_sum = vec![0.0; weights_hidden_output.len()];
    let mut grad_b_output_sum = 0.0;

    for (segment, true_label) in segments.iter().zip(true_labels.iter()) {
        let (output, hidden_activations) = forward_pass(
            segment,
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
        );
        
        let grad_output_layer = find_gradient(&vec![output], &vec![*true_label]);
        let grad_output = grad_output_layer[0];

        // Calculate gradients for output layer
        let grad_w_hidden_output: Vec<f64> = hidden_activations.iter()
            .map(|h| h * grad_output)
            .collect();
        let grad_b_output = grad_output;

        // Calculate error for hidden layer
        let hidden_error_raw: Vec<f64> = weights_hidden_output.iter()
            .map(|w| w * grad_output)
            .collect();
        let hidden_error = vector_elementwise_multiply(&hidden_error_raw, &apply_relu_derivative(&hidden_activations));

        // Calculate gradients for input-hidden layer
        let segment_transposed = matrix_transpose(segment);
        let grad_w_input_hidden = matrix_multiply(&segment_transposed, &vec![hidden_error.clone()]);
        let grad_b_hidden = hidden_error;

        // Accumulate gradients
        matrix_add_inplace(&mut grad_w_input_hidden_sum, &grad_w_input_hidden);
        vector_add_inplace(&mut grad_b_hidden_sum, &grad_b_hidden);
        vector_add_inplace(&mut grad_w_hidden_output_sum, &grad_w_hidden_output);
        grad_b_output_sum += grad_b_output;
    }

    let n = segments.len() as f64;
    (
        matrix_scalar_divide(&grad_w_input_hidden_sum, n),
        vector_scalar_divide(&grad_b_hidden_sum, n),
        vector_scalar_divide(&grad_w_hidden_output_sum, n),
        grad_b_output_sum / n,
    )
}

// Update weights and biases using gradients
pub fn update_weights(
    weights_input_hidden: &mut Vec<Vec<f64>>,
    bias_hidden: &mut Vec<f64>,
    weights_hidden_output: &mut Vec<f64>,
    bias_output: &mut f64,
    grad_w_input_hidden: &Vec<Vec<f64>>,
    grad_b_hidden: &Vec<f64>,
    grad_w_hidden_output: &Vec<f64>,
    grad_b_output: f64,
    learning_rate: f64,
) {
    // Update input-hidden weights
    for i in 0..weights_input_hidden.len() {
        for j in 0..weights_input_hidden[0].len() {
            weights_input_hidden[i][j] -= learning_rate * grad_w_input_hidden[i][j];
        }
    }

    // Update hidden bias
    for i in 0..bias_hidden.len() {
        bias_hidden[i] -= learning_rate * grad_b_hidden[i];
    }

    // Update hidden-output weights
    for i in 0..weights_hidden_output.len() {
        weights_hidden_output[i] -= learning_rate * grad_w_hidden_output[i];
    }

    // Update output bias
    *bias_output -= learning_rate * grad_b_output;
}

// Utility functions for training

// Calculate accuracy for binary classification
pub fn calculate_accuracy(outputs: &Vec<f64>, true_labels: &Vec<f64>) -> f64 {
    let mut correct = 0;
    let total = outputs.len();
    
    for (output, true_label) in outputs.iter().zip(true_labels.iter()) {
        let predicted = if *output > 0.5 { 1.0 } else { 0.0 };
        if predicted == *true_label {
            correct += 1;
        }
    }
    
    correct as f64 / total as f64
}

// Train the neural network for one epoch
pub fn train_epoch(
    segments: &Vec<Vec<Vec<f64>>>,
    true_labels: &Vec<f64>,
    weights_input_hidden: &mut Vec<Vec<f64>>,
    bias_hidden: &mut Vec<f64>,
    weights_hidden_output: &mut Vec<f64>,
    bias_output: &mut f64,
    learning_rate: f64,
) -> f64 {
    // Forward pass to get predictions
    let mut outputs = Vec::new();
    for segment in segments {
        let (output, _) = forward_pass(
            segment,
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            *bias_output,
        );
        outputs.push(output);
    }
    
    // Calculate loss
    let loss = loss_function(&outputs, true_labels);
    
    // Backward pass
    let (grad_w_input_hidden, grad_b_hidden, grad_w_hidden_output, grad_b_output) = backprop(
        segments,
        weights_input_hidden,
        bias_hidden,
        weights_hidden_output,
        *bias_output,
        true_labels,
    );
    
    // Update weights
    update_weights(
        weights_input_hidden,
        bias_hidden,
        weights_hidden_output,
        bias_output,
        &grad_w_input_hidden,
        &grad_b_hidden,
        &grad_w_hidden_output,
        grad_b_output,
        learning_rate,
    );
    
    loss
}

// Predict using the trained model
pub fn predict(
    segment: &Vec<Vec<f64>>,
    weights_input_hidden: &Vec<Vec<f64>>,
    bias_hidden: &Vec<f64>,
    weights_hidden_output: &Vec<f64>,
    bias_output: f64,
) -> f64 {
    let (output, _) = forward_pass(
        segment,
        weights_input_hidden,
        bias_hidden,
        weights_hidden_output,
        bias_output,
    );
    output
}

// Batch prediction
pub fn predict_batch(
    segments: &Vec<Vec<Vec<f64>>>,
    weights_input_hidden: &Vec<Vec<f64>>,
    bias_hidden: &Vec<f64>,
    weights_hidden_output: &Vec<f64>,
    bias_output: f64,
) -> Vec<f64> {
    segments.iter()
        .map(|segment| predict(segment, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output))
        .collect()
}