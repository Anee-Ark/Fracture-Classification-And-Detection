# Fracture Classification and Detection Using Parallel Machine Learning
## Introduction
This project aims to leverage the power of Parallel Machine Learning to enhance fracture detection in X-ray images. The integration of Artificial Intelligence (AI) and Machine Learning (ML) in medical imaging is a transformative innovation in the healthcare field, particularly in trauma care and orthopedic diagnostics. This project, developed for CSYE7105 at Northeastern University, applies advanced ML techniques to significantly improve the accuracy and efficiency of fracture detection while reducing diagnostic time.

## Motivation
With the rising number of X-ray examinations worldwide, the burden on radiologists has increased. This bottleneck leads to delays in diagnosis and treatment, which may adversely affect patient outcomes. Traditional manual review processes are prone to errors and inefficiencies. Our project addresses this challenge by automating the fracture detection process using parallel computing and ML models to identify subtle fracture patterns that might be missed by the human eye.

## Project Goals
The primary goal of this project is to create a parallel machine learning system that:

Automates the detection and classification of bone fractures from X-ray images.
Enhances the speed and accuracy of diagnosis by leveraging parallel processing.
Utilizes GPU acceleration and data parallelism to process large datasets efficiently.
Integrates with convolutional neural networks (CNNs) to improve fracture classification.

## Features
Data Parallelism: Use of multiple CPUs/GPUs to handle large datasets, reducing training and inference times.
Efficient Data Handling: Data preparation with Dask and PyTorch ensures that large image datasets are processed in parallel, speeding up the workflow.
High Accuracy: Achieved an 83% classification accuracy in fracture detection, demonstrating the effectiveness of the model.
CUDA Acceleration: Integrated CUDA for data preprocessing to utilize GPU resources and minimize processing time.
Model Optimization: Implements advanced regularization techniques and optimizes neural network architectures for medical imaging.

# Methodology
## Data Preparation
Data preparation is key in this project. We employed two different approaches for handling X-ray images:

Dask-based Approach: Scales across machines, making it ideal for massive datasets that cannot fit into memory.
PyTorch-based Approach: Best for smaller datasets that fit in memory, offering faster performance due to its direct integration with neural networks.
Model Training
We utilized PyTorch for training convolutional neural networks (CNNs). The training loop includes:

Data loading via PyTorch DataLoader with multiprocessing.
Parallel processing across multiple CPUs/GPUs.
CUDA acceleration for handling computationally expensive operations.

## Parallel Computing
The system employs Distributed Data Parallel (DDP) in PyTorch, distributing the model and data across multiple processors to improve the training speed. We also experimented with hybrid parallelism, combining model parallelism and data parallelism.

# Results and Analysis
Training Time Reduction: Through parallel processing, training time was reduced by 67% compared to traditional sequential methods.
Scalability: Adding more CPUs led to faster processing times, though diminishing returns were observed after 4 CPUs. DDP showed significant performance improvements.
Classification Accuracy: The model achieved an 83% accuracy rate on the test set, illustrating its reliability in identifying and classifying fractures.
Visualizations: Confusion matrices and other visual tools are used to analyze the model’s performance across different fracture types.

# Improvements and Future Work
We identified areas where further improvements can be made:

Data Loading Optimization: Use GPU-accelerated data transformation techniques for faster data loading.
Hybrid Parallelism: Combine model and data parallelism to reduce communication overhead.
Efficient Architectures: Explore architectures like MobileNets for faster training and inference.
Advanced Regularization: Employ more sophisticated regularization techniques to improve model generalization.

# Conclusion
This project demonstrates the transformative potential of parallel computing in medical imaging. By automating the process of fracture detection using AI and ML, we have reduced diagnostic time and improved accuracy, thereby setting new standards for healthcare diagnostics.

