# ECGNN

## Arrhythmia Detection Neural Network (Python & Rust)

 **From-scratch neural network for arrhythmia classification**  
Prototype in Python to a High-performance reimplementation in Rust using no libraries

---

## Project Overview

This project develops a basic neural network from scratch to detect arrhythmias from ECG signals. The goal is to:

- Prototype and validate the logic in **pure Python** (minimal/no external libraries)
- Rebuild the system in **Rust** for safe, efficient, real-world performance

---

## Planned Architecture

### Python Prototype
- Manual matrix operations or minimal NumPy (if used)
- Basic feedforward neural network
- Small-scale testing with ECG windows

### Rust Reimplementation
- Full rewrite in Rust
- Manual matrix/vector math or use of `nalgebra`
- Memory-safe, performant architecture suitable for real datasets

---

## To-Do List

### Python Prototype
- [X] Project structure setup
- [X] Data loading functions - load_data and load_all_data
- [X] Signal preprocessing (normalization, segmentation)
- [X] Forward pass implementation
- [ ] Backpropagation logic
- [ ] Loss calculation
- [ ] Basic training loop
- [ ] Toy dataset testing
- [ ] Evaluate classification accuracy

### Rust Reimplementation
- [ ] Project structure setup
- [ ] Data handling module
- [ ] Linear algebra module (`nalgebra` optional)
- [ ] Neural network core
- [ ] Loss functions
- [ ] Backpropagation
- [ ] Training & evaluation
- [ ] Performance optimization

---

## Dataset

- Target dataset: **MIT-BIH Arrhythmia Database**
- Initial testing with synthetic or small sample ECG segments

---
