# project13_multi_class_classification

# Restaurant Customer Potential Classification

## Overview

This project explores **multiclass classification using PyTorch** by building a feedforward neural network to predict restaurant customer potential.

The main objective was to gain practical experience with neural networks and understand how they learn to classify data into multiple categories.

The project focuses particularly on understanding `CrossEntropyLoss()`, class indices, forward propagation, backpropagation and model optimisation.

The model predicts one of three customer potential categories:

- `0` – Non-Potential Customer
- `1` – Moderate-Potential Customer
- `2` – High-Potential Customer

## Python Modules & Libraries

- `pandas` – Data manipulation and cleaning.
- `numpy` – Numerical operations.
- `matplotlib.pyplot` – Data visualisation.
- `sklearn.model_selection` – Splitting data into training, development and test sets.
- `sklearn.preprocessing` – Feature scaling and encoding.
- `sklearn.feature_selection` – Variance-based feature selection.
- `torch` – Tensor operations and neural network development.
- `torch.nn` – Neural network layers and loss functions.
- `torch.optim` – Model optimisation.
- `torch.utils.data` – Dataset and DataLoader management.

## Data Preprocessing

Before training the neural network, the dataset underwent several preprocessing steps:

- Removed unnecessary columns.
- Cleaned inconsistent categorical values.
- Standardised category names.
- Converted categorical features into numerical representations using one-hot encoding.
- Converted string-based numerical values into numerical data.
- Corrected the delivery-to-dine-in ratio representation (`70/30` to `2.33`).
- Handled missing values using `dropna()`.

## Dataset Preparation

The dataset was divided into three subsets:

- 80% Training data
- 10% Development data
- 10% Testing data

Used `train_test_split()` with `random_state=42` to make the data split reproducible.

Feature selection and standardisation were fitted on the training data and then applied to the development and test datasets to prevent data leakage.

## Neural Network Architecture

The project uses a feedforward neural network built with PyTorch.

| Layer | Neurons | Activation |
|---|---:|---|
| Input layer | Number of input features | — |
| Hidden layer 1 | 12 | ReLU |
| Hidden layer 2 | 6 | ReLU |
| Output layer | 3 | linear activation |

The output layer contains three neurons, representing the three possible customer potential classes.

## Model Configuration

| Component | Configuration |
|---|---|
| Framework | PyTorch |
| Model type | Feedforward Neural Network |
| Loss function | CrossEntropyLoss |
| Optimiser | AdamW |
| Learning rate | 0.001 |
| Training epochs | 1000 |
| Batch size | 50 |
| Hidden layer activation | ReLU |
| Output classes | 3 |

## Understanding Multiclass Classification

A major learning objective of this project was understanding how neural networks perform multiclass classification.

### CrossEntropyLoss

Used `nn.CrossEntropyLoss()` to calculate the difference between the model's predicted logits and the actual target classes.

The target labels are represented as integer class indices:

- Loss = 0
- Draw = 1
- Win = 2

For this project, the three customer potential categories are used instead.

The target tensors use `torch.long` because CrossEntropyLoss expects integer class indices.

The output layer produces three logits, and `CrossEntropyLoss` internally applies the appropriate log-softmax and negative log-likelihood calculations.

### Forward Propagation

The input features pass through the network's hidden layers before producing three output logits.

### Backpropagation

The calculated loss is used to determine gradients for the model's parameters.

### Optimisation

AdamW updates the model parameters using the calculated gradients to minimise the loss.

## What I Learned

Through this project, I gained practical experience with:

- Multiclass classification.
- Building neural networks using PyTorch.
- Understanding `CrossEntropyLoss()`.
- Understanding logits and class indices.
- Understanding why multiclass targets use `torch.long`.
- Forward propagation.
- Backpropagation and automatic gradient calculation.
- Optimisers and parameter updates.
- Training, development and test datasets.
- Feature selection and standardisation.
- Evaluating neural network predictions.
- Identifying and investigating overfitting.


## Project Status

**Experimental Learning Project**

This project is part of my ongoing journey in machine learning and deep learning.

The primary focus was learning and implementing multiclass classification with PyTorch, rather than developing a production-ready prediction system.

## Author

**Kamokgelo Mdluli**

Aspiring Deep Learning Engineer

