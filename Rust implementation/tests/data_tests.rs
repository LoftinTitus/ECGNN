use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use ecgnn::data::*;

// Helper function to create test CSV files
fn create_test_csv(path: &str, content: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

// Helper function to create test directory with unique name
fn setup_test_directory() -> std::io::Result<String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let thread_id = thread::current().id();
    let test_dir = format!("test_data_{}_{:?}", timestamp, thread_id);
    
    // Remove directory if it exists, then create it fresh
    if Path::new(&test_dir).exists() {
        fs::remove_dir_all(&test_dir)?;
    }
    fs::create_dir(&test_dir)?;
    
    // Create test CSV files
    create_test_csv(&format!("{}/data1.csv", test_dir), 
        "time,value1,value2\n\
         0.0,1.0,2.0\n\
         1.0,2.0,4.0\n\
         2.0,3.0,6.0\n\
         3.0,4.0,8.0\n"
    )?;
    
    create_test_csv(&format!("{}/data2.csv", test_dir),
        "time,value1,value2\n\
         4.0,5.0,10.0\n\
         5.0,6.0,12.0\n\
         6.0,7.0,14.0\n"
    )?;
    
    // Create a non-CSV file to test filtering
    create_test_csv(&format!("{}/not_csv.txt", test_dir), "This is not a CSV file")?;
    
    Ok(test_dir)
}

// Helper function to clean up test directory
fn cleanup_test_directory(test_dir: &str) -> std::io::Result<()> {
    if Path::new(test_dir).exists() {
        fs::remove_dir_all(test_dir)?;
    }
    Ok(())
}

#[cfg(test)]
mod data_tests {
    use super::*;

    #[test]
    fn test_load_csv_success() {
        let test_dir = setup_test_directory().expect("Failed to setup test directory");
        
        let result = load_csv(&format!("{}/data1.csv", test_dir));
        assert!(result.is_ok());
        
        let data = result.unwrap();
        assert_eq!(data.len(), 4); // 4 rows of data (excluding header)
        assert_eq!(data[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(data[1], vec![1.0, 2.0, 4.0]);
        assert_eq!(data[2], vec![2.0, 3.0, 6.0]);
        assert_eq!(data[3], vec![3.0, 4.0, 8.0]);
        
        cleanup_test_directory(&test_dir).expect("Failed to cleanup test directory");
    }

    #[test]
    fn test_load_csv_nonexistent_file() {
        let result = load_csv("nonexistent.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_all_data() {
        let test_dir = setup_test_directory().expect("Failed to setup test directory");
        
        let result = load_all_data(&test_dir);
        assert!(result.is_ok());
        
        let data = result.unwrap();
        assert_eq!(data.len(), 7); // 4 rows from data1.csv + 3 rows from data2.csv
        
        // Check that data from both files is included
        assert!(data.contains(&vec![0.0, 1.0, 2.0])); // from data1.csv
        assert!(data.contains(&vec![4.0, 5.0, 10.0])); // from data2.csv
        
        cleanup_test_directory(&test_dir).expect("Failed to cleanup test directory");
    }

    #[test]
    fn test_load_all_data_empty_directory() {
        // Create empty directory
        let empty_dir = "empty_test_dir";
        fs::create_dir(empty_dir).expect("Failed to create empty directory");
        
        let result = load_all_data(empty_dir);
        assert!(result.is_ok());
        
        let data = result.unwrap();
        assert_eq!(data.len(), 0);
        
        fs::remove_dir(empty_dir).expect("Failed to remove empty directory");
    }

    #[test]
    fn test_data_scaling_normal_case() {
        let test_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        
        let scaled = data_scaling(&test_data);
        
        // Check dimensions
        assert_eq!(scaled.len(), 3);
        assert_eq!(scaled[0].len(), 3);
        
        // First column should remain unchanged
        assert_eq!(scaled[0][0], 1.0);
        assert_eq!(scaled[1][0], 2.0);
        assert_eq!(scaled[2][0], 3.0);
        
        // Check that columns 2 and 3 are scaled (mean should be approximately 0)
        let col2_mean: f64 = scaled.iter().map(|row| row[1]).sum::<f64>() / scaled.len() as f64;
        let col3_mean: f64 = scaled.iter().map(|row| row[2]).sum::<f64>() / scaled.len() as f64;
        
        assert!((col2_mean).abs() < 1e-10); // Should be very close to 0
        assert!((col3_mean).abs() < 1e-10); // Should be very close to 0
    }

    #[test]
    fn test_data_scaling_zero_variance() {
        let test_data = vec![
            vec![1.0, 5.0, 3.0],
            vec![2.0, 5.0, 6.0],
            vec![3.0, 5.0, 9.0],
        ];
        
        let scaled = data_scaling(&test_data);
        
        // Column 2 has zero variance, so it should be scaled to 0
        assert_eq!(scaled[0][1], 0.0);
        assert_eq!(scaled[1][1], 0.0);
        assert_eq!(scaled[2][1], 0.0);
    }

    #[test]
    fn test_data_scaling_empty_data() {
        let test_data: Vec<Vec<f64>> = vec![];
        let scaled = data_scaling(&test_data);
        assert_eq!(scaled.len(), 0);
    }

    #[test]
    fn test_data_segmentation_normal_case() {
        let test_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
            vec![4.0, 8.0, 12.0],
            vec![5.0, 10.0, 15.0],
        ];
        
        let segments = data_segmentation(&test_data, 2);
        
        // Should have 2 complete segments of length 2
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 2);
        assert_eq!(segments[1].len(), 2);
        
        // Check segment contents
        assert_eq!(segments[0][0], vec![1.0, 2.0, 3.0]);
        assert_eq!(segments[0][1], vec![2.0, 4.0, 6.0]);
        assert_eq!(segments[1][0], vec![3.0, 6.0, 9.0]);
        assert_eq!(segments[1][1], vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_data_segmentation_incomplete_segment() {
        let test_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        
        let segments = data_segmentation(&test_data, 2);
        
        // Should have 1 complete segment (incomplete segment is discarded)
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].len(), 2);
    }

    #[test]
    fn test_data_segmentation_empty_data() {
        let test_data: Vec<Vec<f64>> = vec![];
        let segments = data_segmentation(&test_data, 2);
        assert_eq!(segments.len(), 0);
    }

    #[test]
    fn test_data_segmentation_segment_size_larger_than_data() {
        let test_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
        ];
        
        let segments = data_segmentation(&test_data, 5);
        
        // No complete segments should be created
        assert_eq!(segments.len(), 0);
    }

    #[test]
    fn test_end_to_end_workflow() {
        let test_dir = setup_test_directory().expect("Failed to setup test directory");
        
        // Load data
        let data = load_all_data(&test_dir).expect("Failed to load data");
        println!("Loaded {} rows of data", data.len());
        
        // Scale data
        let scaled_data = data_scaling(&data);
        println!("Scaled data has {} rows", scaled_data.len());
        
        // Segment data
        let segments = data_segmentation(&scaled_data, 3);
        println!("Created {} segments of length 3", segments.len());
        
        // Basic assertions
        assert!(!data.is_empty());
        assert_eq!(scaled_data.len(), data.len());
        assert!(!segments.is_empty());
        
        cleanup_test_directory(&test_dir).expect("Failed to cleanup test directory");
    }
}

// Integration test that can be run manually
#[test]
fn integration_test_with_real_data() {
    // This test assumes you have real CSV files in a 'real_data' directory
    // Comment out or modify this test based on your actual data structure
    
    if Path::new("real_data").exists() {
        println!("Running integration test with real data...");
        
        match load_all_data("real_data") {
            Ok(data) => {
                println!("Successfully loaded {} rows of real data", data.len());
                
                if !data.is_empty() {
                    let scaled = data_scaling(&data);
                    let segments = data_segmentation(&scaled, 10);
                    println!("Created {} segments from real data", segments.len());
                }
            }
            Err(e) => {
                println!("Error loading real data: {}", e);
            }
        }
    } else {
        println!("No real_data directory found, skipping integration test");
    }
}
