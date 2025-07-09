pub fn matrix_multiply(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let rows_b = b.len();
    let cols_b = b[0].len();

    if cols_a != rows_b {
        panic!("Matrix dimensions do not match for multiplication");
    }

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

pub fn matrix_transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j];
        }
    }

    result
}

pub fn matrix_add(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = a[0].len();

    if rows != b.len() || cols != b[0].len() {
        panic!("Matrix dimensions do not match for addition");
    }

    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = a[i][j] + b[i][j];
        }
    }

    result
}

pub fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    if a.len() != b.len() {
        panic!("Vectors must be of the same length for dot product");
    }

    let mut result = 0.0;

    for i in 0..a.len() {
        result += a[i] * b[i];
    }

    result
}

pub fn scalar_multiply(matrix: &Vec<Vec<f64>>, scalar: f64) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = matrix[i][j] * scalar;
        }
    }

    result
}

pub fn elementwise_multiply(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = a[0].len();

    if rows != b.len() || cols != b[0].len() {
        panic!("Matrix dimensions do not match for element-wise multiplication");
    }

    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = a[i][j] * b[i][j];
        }
    }

    result
}

pub fn vector_add(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    if a.len() != b.len() {
        panic!("Vectors must be of the same length for addition");
    }
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Matrix addition in-place - modifies first matrix
pub fn matrix_add_inplace(a: &mut Vec<Vec<f64>>, b: &Vec<Vec<f64>>) {
    if a.len() != b.len() || a[0].len() != b[0].len() {
        panic!("Matrices must be of the same dimensions for addition");
    }
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            a[i][j] += b[i][j];
        }
    }
}

/// Vector addition in-place - modifies first vector
pub fn vector_add_inplace(a: &mut Vec<f64>, b: &Vec<f64>) {
    if a.len() != b.len() {
        panic!("Vectors must be of the same length for addition");
    }
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

pub fn vector_scalar_divide(vector: &Vec<f64>, scalar: f64) -> Vec<f64> {
    vector.iter().map(|x| x / scalar).collect()
}

pub fn matrix_scalar_divide(matrix: &Vec<Vec<f64>>, scalar: f64) -> Vec<Vec<f64>> {
    matrix.iter()
        .map(|row| row.iter().map(|x| x / scalar).collect())
        .collect()
}

pub fn vector_elementwise_multiply(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    if a.len() != b.len() {
        panic!("Vectors must be of the same length for element-wise multiplication");
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Initialize gradient matrix with same dimensions as input, filled with zeros
pub fn initialize_gradient_matrix_like(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    vec![vec![0.0; matrix[0].len()]; matrix.len()]
}

/// Initialize gradient vector with same length as input, filled with zeros
pub fn initialize_gradient_vector_like(vector: &Vec<f64>) -> Vec<f64> {
    vec![0.0; vector.len()]
}