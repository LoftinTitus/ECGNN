use std::fs::{self, File};
use std::io::{BufRead, BufReader};

/// Load CSV data from a file
/// Returns a vector of vectors where each inner vector represents a row of float values
pub fn load_csv(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        
        // Skip header (first line)
        if index == 0 {
            continue;
        }
        
        if !line.trim().is_empty() {
            let values: Result<Vec<f64>, _> = line
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect();
            
            match values {
                Ok(float_values) => data.push(float_values),
                Err(e) => eprintln!("Error parsing line {}: {}", index + 1, e),
            }
        }
    }
    
    Ok(data)
}

/// Load all CSV files from a folder
/// Returns a vector containing all data from all CSV files in the folder
pub fn load_all_data(folder_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let mut all_data = Vec::new();
    
    for entry in fs::read_dir(folder_path)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "csv" {
                    if let Some(path_str) = path.to_str() {
                        match load_csv(path_str) {
                            Ok(mut file_data) => all_data.append(&mut file_data),
                            Err(e) => eprintln!("Error loading file {:?}: {}", path, e),
                        }
                    }
                }
            }
        }
    }
    
    Ok(all_data)
}

/// Scale data by normalizing columns 2 and 3 (indices 1 and 2)
/// Returns a new vector with scaled data
pub fn data_scaling(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return Vec::new();
    }
    
    // Extract columns 2 and 3 (indices 1 and 2)
    let col2: Vec<f64> = data.iter().map(|row| row[1]).collect();
    let col3: Vec<f64> = data.iter().map(|row| row[2]).collect();
    
    // Calculate means
    let mean2 = col2.iter().sum::<f64>() / col2.len() as f64;
    let mean3 = col3.iter().sum::<f64>() / col3.len() as f64;
    
    // Calculate variances
    let variance2 = col2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / col2.len() as f64;
    let variance3 = col3.iter().map(|x| (x - mean3).powi(2)).sum::<f64>() / col3.len() as f64;
    
    // Create scaled dataset
    let mut scaled_dataset = Vec::new();
    
    for row in data {
        let scaled_row2 = if variance2 != 0.0 {
            (row[1] - mean2) / variance2.sqrt()
        } else {
            0.0
        };
        
        let scaled_row3 = if variance3 != 0.0 {
            (row[2] - mean3) / variance3.sqrt()
        } else {
            0.0
        };
        
        scaled_dataset.push(vec![row[0], scaled_row2, scaled_row3]);
    }
    
    scaled_dataset
}

/// Segment data into chunks of specified length
/// Returns a vector of segments, each containing segment_length rows
pub fn data_segmentation(scaled_dataset: &[Vec<f64>], segment_length: usize) -> Vec<Vec<Vec<f64>>> {
    let mut segments = Vec::new();
    
    for i in (0..scaled_dataset.len()).step_by(segment_length) {
        let end_index = std::cmp::min(i + segment_length, scaled_dataset.len());
        let segment = scaled_dataset[i..end_index].to_vec();
        
        // Only add segments that are exactly the specified length
        if segment.len() == segment_length {
            segments.push(segment);
        }
    }
    
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_scaling() {
        let test_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        
        let scaled = data_scaling(&test_data);
        
        // Check that we get the expected number of rows
        assert_eq!(scaled.len(), 3);
        
        // Check that each row has 3 columns
        for row in &scaled {
            assert_eq!(row.len(), 3);
        }
        
        // First column should remain unchanged
        assert_eq!(scaled[0][0], 1.0);
        assert_eq!(scaled[1][0], 2.0);
        assert_eq!(scaled[2][0], 3.0);
    }

    #[test]
    fn test_data_segmentation() {
        let test_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
            vec![4.0, 8.0, 12.0],
            vec![5.0, 10.0, 15.0],
        ];
        
        let segments = data_segmentation(&test_data, 2);
        
        // Should have 2 segments of length 2 each
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 2);
        assert_eq!(segments[1].len(), 2);
        
        // Check first segment
        assert_eq!(segments[0][0], vec![1.0, 2.0, 3.0]);
        assert_eq!(segments[0][1], vec![2.0, 4.0, 6.0]);
        
        // Check second segment
        assert_eq!(segments[1][0], vec![3.0, 6.0, 9.0]);
        assert_eq!(segments[1][1], vec![4.0, 8.0, 12.0]);
    }
}