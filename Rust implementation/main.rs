use ecgnn::data::{load_all_data, data_scaling, data_segmentation};
use ecgnn::brains::*;
use std::io::{self, Write};

/// Display a progress bar
fn display_progress(current: usize, total: usize, loss: f64, accuracy: f64) {
    let progress = (current as f64 / total as f64) * 100.0;
    let bar_length = 50;
    let filled_length = ((progress / 100.0) * bar_length as f64) as usize;
    
    let bar = "=".repeat(filled_length) + &">".repeat(if filled_length < bar_length { 1 } else { 0 }) + &" ".repeat(bar_length - filled_length - if filled_length < bar_length { 1 } else { 0 });
    
    let acc_display = if accuracy == 0.0 { 
        "Calculating...".to_string() 
    } else { 
        format!("{:.1}%", accuracy * 100.0) 
    };
    
    print!("\rEpoch {}/{} [{}] {:.1}% | Loss: {:.4} | Acc: {}", 
           current + 1, total, bar, progress, loss, acc_display);
    io::stdout().flush().unwrap();
}

/// Display training statistics
fn display_training_stats(epoch: usize, total_epochs: usize, loss: f64, accuracy: f64, elapsed_time: f64) {
    let remaining_time = if epoch > 0 {
        elapsed_time * (total_epochs - epoch - 1) as f64 / epoch as f64
    } else {
        0.0
    };
    
    println!("\n Epoch {}/{} Complete:", epoch + 1, total_epochs);
    println!("   Loss: {:.4}", loss);
    println!("   Accuracy: {:.2}%", accuracy * 100.0);
    println!("   Time: {:.2}s | Est. remaining: {:.2}s", elapsed_time, remaining_time);
    println!("   {}", "─".repeat(60));
}

fn main() {
    println!("=== ECGNN - ECG Arrhythmia Detection Neural Network ===");
    
    // Try to load your actual ECG data
    let data_folder = "/Users/tyloftin/Downloads/MIT Data";
    let data = match load_all_data(data_folder) {
        Ok(data) => {
            println!("Loaded {} data points from {}", data.len(), data_folder);
            data
        },
        Err(e) => {
            eprintln!("Error loading data from {}: {}", data_folder, e);
            println!("Falling back to synthetic data for demonstration...");
            generate_synthetic_data()
        }
    };
    
    // Scale the data
    let scaled_data = data_scaling(&data);
    println!("Data scaled successfully");
    
    // Display some dataset statistics
    if !data.is_empty() {
        println!("Dataset info:");
        println!("  Total samples: {}", data.len());
        println!("  Features per sample: {}", data[0].len());
        println!("  Dataset size: {:.2} MB", (data.len() * data[0].len() * 8) as f64 / (1024.0 * 1024.0));
    }
    
    // Create segments
    let segment_size = 250; // Increased for real ECG data (typically better for heart rhythm analysis)
    let segments = data_segmentation(&scaled_data, segment_size);
    println!("Created {} segments of size {}", segments.len(), segment_size);
    
    if segments.is_empty() {
        eprintln!("No segments created");
        return;
    }
    
    // Convert segments to flattened feature vectors
    let flattened_segments: Vec<Vec<f64>> = segments.iter()
        .map(|segment| {
            segment.iter().flatten().cloned().collect()
        })
        .collect();
    
    let num_features = flattened_segments[0].len();
    println!("Each segment has {} features", num_features);
    
    // Create labels - for real ECG data, you'd load actual labels
    // For now, we'll create synthetic labels based on data characteristics
    let labels = create_ecg_labels(&flattened_segments);
    
    // Split data (80/20)
    let split_index = (flattened_segments.len() as f64 * 0.8) as usize;
    let train_segments = flattened_segments[..split_index].to_vec();
    let test_segments = flattened_segments[split_index..].to_vec();
    let train_labels = labels[..split_index].to_vec();
    let test_labels = labels[split_index..].to_vec();
    
    println!("Training set: {} samples", train_segments.len());
    println!("Test set: {} samples", test_segments.len());
    
    // Initialize neural network
    let num_hidden = 64;
    let (mut weights_input_hidden, mut bias_hidden) = initialize_weights_and_bias(num_features, num_hidden);
    let mut weights_hidden_output = vec![0.01; num_hidden];
    let mut bias_output = 0.01;
    
    println!("
Initialized neural network:");
    println!("  Input features: {}", num_features);
    println!("  Hidden neurons: {}", num_hidden);
    
    // Convert to format expected by neural network
    let train_segments_nn: Vec<Vec<Vec<f64>>> = train_segments.iter()
        .map(|segment| vec![segment.clone()])
        .collect();
    let test_segments_nn: Vec<Vec<Vec<f64>>> = test_segments.iter()
        .map(|segment| vec![segment.clone()])
        .collect();
    
    // Training loop
    println!("\nStarting training...");
    let epochs = 50;
    let learning_rate = 0.01;
    
    let start_time = std::time::Instant::now();
    
    for epoch in 0..epochs {
        let epoch_start = std::time::Instant::now();
        
        let loss = train_epoch(
            &train_segments_nn,
            &train_labels,
            &mut weights_input_hidden,
            &mut bias_hidden,
            &mut weights_hidden_output,
            &mut bias_output,
            learning_rate,
        );
        
        // Show progress immediately after each epoch
        display_progress(epoch, epochs, loss, 0.0); // Use 0.0 for accuracy to avoid slow calculation
        
        // Calculate accuracy and show detailed stats only every 10 epochs
        if epoch % 2 == 0 || epoch == epochs - 1 {
            let train_predictions = predict_batch(
                &train_segments_nn,
                &weights_input_hidden,
                &bias_hidden,
                &weights_hidden_output,
                bias_output,
            );
            let train_accuracy = calculate_accuracy(&train_predictions, &train_labels);
            
            let elapsed = epoch_start.elapsed().as_secs_f64();
            display_training_stats(epoch, epochs, loss, train_accuracy, elapsed);
        }
    }
    
    let total_training_time = start_time.elapsed().as_secs_f64();
    println!("\n Training completed in {:.2} seconds!", total_training_time);
    
    // Test the model
    println!("
Testing the model...");
    let test_predictions = predict_batch(
        &test_segments_nn,
        &weights_input_hidden,
        &bias_hidden,
        &weights_hidden_output,
        bias_output,
    );
    
    let test_accuracy = calculate_accuracy(&test_predictions, &test_labels);
    let test_loss = loss_function(&test_predictions, &test_labels);
    
    println!("Test Results:");
    println!("  Test Loss: {:.4}", test_loss);
    println!("  Test Accuracy: {:.2}%", test_accuracy * 100.0);
    
    println!("
=== Training Complete ===");
}

fn generate_synthetic_data() -> Vec<Vec<f64>> {
    let mut data = Vec::new();
    
    for i in 0..1000 {
        let mut sample = Vec::new();
        
        for j in 0..200 {
            let base_signal = (j as f64 * 0.1).sin() * 0.5;
            let noise = (i as f64 * 0.01 + j as f64 * 0.02).sin() * 0.1;
            let heart_beat = if j % 50 == 0 { 1.0 } else { 0.0 };
            
            sample.push(base_signal + noise + heart_beat);
        }
        
        data.push(sample);
    }
    
    data
}

fn create_synthetic_labels(num_samples: usize) -> Vec<f64> {
    let mut labels = Vec::new();
    
    for i in 0..num_samples {
        let label = if i % 3 == 0 { 1.0 } else { 0.0 };
        labels.push(label);
    }
    
    labels
}

/// Create labels based on ECG data characteristics
fn create_ecg_labels(segments: &[Vec<f64>]) -> Vec<f64> {
    let mut labels = Vec::new();
    
    for segment in segments {
        // Simple heuristic based on signal characteristics
        let mean = segment.iter().sum::<f64>() / segment.len() as f64;
        let variance = segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / segment.len() as f64;
        let std_dev = variance.sqrt();
        
        // Label as arrhythmia if standard deviation is unusually high or low
        let label = if std_dev > 1.5 || std_dev < 0.3 {
            1.0 // Arrhythmia
        } else {
            0.0 // Normal
        };
        
        labels.push(label);
    }
    
    labels
}
