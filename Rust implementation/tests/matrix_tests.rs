#[cfg(test)]
mod tests {
    use ecgnn::matrix_math::*; 

    #[test]
    fn test_matrix_multiply() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = matrix_multiply(&a, &b);
        let expected = vec![vec![19.0, 22.0], vec![43.0, 50.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiply_single_element() {
        let a = vec![vec![2.0]];
        let b = vec![vec![3.0]];
        let result = matrix_multiply(&a, &b);
        let expected = vec![vec![6.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiply_identity() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = matrix_multiply(&a, &identity);
        assert_eq!(result, a);
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions do not match for multiplication")]
    fn test_matrix_multiply_dimension_mismatch() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.0], vec![2.0], vec![3.0]];
        matrix_multiply(&a, &b);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = matrix_transpose(&matrix);
        let expected = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_transpose_square() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = matrix_transpose(&matrix);
        let expected = vec![vec![1.0, 3.0], vec![2.0, 4.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_transpose_single_row() {
        let matrix = vec![vec![1.0, 2.0, 3.0]];
        let result = matrix_transpose(&matrix);
        let expected = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_add() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = matrix_add(&a, &b);
        let expected = vec![vec![6.0, 8.0], vec![10.0, 12.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_add_zeros() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let zeros = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = matrix_add(&a, &zeros);
        assert_eq!(result, a);
    }

    #[test]
    fn test_matrix_add_negative() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
        let result = matrix_add(&a, &b);
        let expected = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions do not match for addition")]
    fn test_matrix_add_dimension_mismatch() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.0], vec![2.0]];
        matrix_add(&a, &b);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_dot_product_zeros() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let result = dot_product(&a, &b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_product_single_element() {
        let a = vec![5.0];
        let b = vec![3.0];
        let result = dot_product(&a, &b);
        assert_eq!(result, 15.0);
    }

    #[test]
    #[should_panic(expected = "Vectors must be of the same length for dot product")]
    fn test_dot_product_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        dot_product(&a, &b);
    }

    #[test]
    fn test_scalar_multiply() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = scalar_multiply(&matrix, 2.0);
        let expected = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiply_zero() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = scalar_multiply(&matrix, 0.0);
        let expected = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiply_negative() {
        let matrix = vec![vec![1.0, -2.0], vec![3.0, -4.0]];
        let result = scalar_multiply(&matrix, -1.0);
        let expected = vec![vec![-1.0, 2.0], vec![-3.0, 4.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_elementwise_multiply() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
        let result = elementwise_multiply(&a, &b);
        let expected = vec![vec![2.0, 6.0], vec![12.0, 20.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_elementwise_multiply_zeros() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let result = elementwise_multiply(&a, &b);
        let expected = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_elementwise_multiply_ones() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let result = elementwise_multiply(&a, &b);
        assert_eq!(result, a);
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions do not match for element-wise multiplication")]
    fn test_elementwise_multiply_dimension_mismatch() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.0], vec![2.0]];
        elementwise_multiply(&a, &b);
    }

    #[test]
    fn test_all_functions() {
        println!("Testing all matrix functions...");
        
        // Test matrix_multiply
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = matrix_multiply(&a, &b);
        let expected = vec![vec![19.0, 22.0], vec![43.0, 50.0]];
        if result != expected {
            panic!("matrix_multiply is not working correctly");
        }

        // Test matrix_transpose
        let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = matrix_transpose(&matrix);
        let expected = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        if result != expected {
            panic!("matrix_transpose is not working correctly");
        }

        // Test matrix_add
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = matrix_add(&a, &b);
        let expected = vec![vec![6.0, 8.0], vec![10.0, 12.0]];
        if result != expected {
            panic!("matrix_add is not working correctly");
        }

        // Test dot_product
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        if result != 32.0 {
            panic!("dot_product is not working correctly");
        }

        // Test scalar_multiply
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = scalar_multiply(&matrix, 2.0);
        let expected = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        if result != expected {
            panic!("scalar_multiply is not working correctly");
        }

        // Test elementwise_multiply
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
        let result = elementwise_multiply(&a, &b);
        let expected = vec![vec![2.0, 6.0], vec![12.0, 20.0]];
        if result != expected {
            panic!("elementwise_multiply is not working correctly");
        }

        println!("All functions are working correctly - all good!");
    }
}