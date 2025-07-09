// Example demonstrating how to use the data handling functions
use ecgnn::data::*;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Data Handling Example ===\n");
    
    // Create a sample CSV file for demonstration
    create_sample_data()?;
    
    // Test loading a single CSV file
    println!("1. Loading single CSV file:");
    match load_csv("example_data.csv") {
        Ok(data) => {
            println!("   Loaded {} rows of data", data.len());
            println!("   First row: {:?}", data[0]);
        }
        Err(e) => println!("   Error: {}", e),
    }
    
    // Test loading all data from a directory
    println!("\n2. Loading all CSV files from directory:");
    match load_all_data("./") {
        Ok(data) => {
            println!("   Loaded {} total rows from all CSV files", data.len());
            if !data.is_empty() {
                println!("   Sample data: {:?}", &data[0..std::cmp::min(3, data.len())]);
            }
        }
        Err(e) => println!("   Error: {}", e),
    }
    
    // Test data scaling
    println!("\n3. Data scaling:");
    let sample_data = vec![
        vec![1.0, 10.0, 100.0],
        vec![2.0, 20.0, 200.0],
        vec![3.0, 30.0, 300.0],
        vec![4.0, 40.0, 400.0],
    ];
    
    println!("   Original data: {:?}", sample_data);
    let scaled_data = data_scaling(&sample_data);
    println!("   Scaled data: {:?}", scaled_data);
    
    // Verify that columns 2 and 3 have mean ≈ 0
    let col2_mean: f64 = scaled_data.iter().map(|row| row[1]).sum::<f64>() / scaled_data.len() as f64;
    let col3_mean: f64 = scaled_data.iter().map(|row| row[2]).sum::<f64>() / scaled_data.len() as f64;
    println!("   Column 2 mean after scaling: {:.10}", col2_mean);
    println!("   Column 3 mean after scaling: {:.10}", col3_mean);
    
    // Test data segmentation
    println!("\n4. Data segmentation:");
    let segments = data_segmentation(&scaled_data, 2);
    println!("   Created {} segments of length 2", segments.len());
    for (i, segment) in segments.iter().enumerate() {
        println!("   Segment {}: {:?}", i + 1, segment);
    }
    
    // Test with different segment sizes
    println!("\n5. Different segment sizes:");
    let large_data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64, (i * 3) as f64]).collect();
    
    for &size in &[3, 4, 5] {
        let segments = data_segmentation(&large_data, size);
        println!("   Segment size {}: {} segments", size, segments.len());
    }
    
    // Complete workflow example
    println!("\n6. Complete workflow example:");
    println!("   Loading -> Scaling -> Segmenting");
    
    if let Ok(raw_data) = load_csv("example_data.csv") {
        let scaled = data_scaling(&raw_data);
        let segments = data_segmentation(&scaled, 3);
        
        println!("   Raw data: {} rows", raw_data.len());
        println!("   Scaled data: {} rows", scaled.len());
        println!("   Segments: {} segments of length 3", segments.len());
        
        if !segments.is_empty() {
            println!("   First segment: {:?}", segments[0]);
        }
    }
    
    // Cleanup
    cleanup_sample_data()?;
    
    println!("\n=== Example completed successfully! ===");
    Ok(())
}

fn create_sample_data() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create("example_data.csv")?;
    writeln!(file, "time,sensor1,sensor2")?;
    writeln!(file, "0.0,1.5,2.1")?;
    writeln!(file, "0.1,1.8,2.4")?;
    writeln!(file, "0.2,2.1,2.8")?;
    writeln!(file, "0.3,1.9,2.6")?;
    writeln!(file, "0.4,2.3,3.1")?;
    writeln!(file, "0.5,2.0,2.9")?;
    writeln!(file, "0.6,2.5,3.3")?;
    writeln!(file, "0.7,2.2,3.0")?;
    writeln!(file, "0.8,2.8,3.6")?;
    writeln!(file, "0.9,2.6,3.4")?;
    
    Ok(())
}

fn cleanup_sample_data() -> Result<(), Box<dyn std::error::Error>> {
    if std::path::Path::new("example_data.csv").exists() {
        fs::remove_file("example_data.csv")?;
    }
    Ok(())
}
