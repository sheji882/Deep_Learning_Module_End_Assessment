# Deep_Learning_Module_End_Assessment
Objective: It is required to model the progression of diabetes using the available independent variables. This model will help healthcare professionals understand how different factors influence the progression of diabetes and potentially aid in designing better treatment plans and preventive measures. The model will provide insights into the dynamics of diabetes progression in patients.

Dataset: Use the Diabetes dataset available in the sklearn library.
Overview
This project implements and optimizes a feedforward neural network to predict a continuous target variable from tabular input features. The goal is to minimize Mean Absolute Error (MAE) while preventing overfitting. Techniques like Dropout, L2 regularization, BatchNormalization, and learning rate scheduling are employed to improve generalization.

Features

Fully connected feedforward network with multiple hidden layers.

L2 regularization and Dropout to prevent overfitting.

BatchNormalization to stabilize training.

Learning rate scheduling via ReduceLROnPlateau.

EarlyStopping to restore the best model based on validation MAE.

Optional feature engineering using polynomial features.

Model Architecture

Input: Number of features in dataset (X_scaled.shape[1]).

Hidden layers:

Dense(96, ReLU) + L2(0.0004) + BatchNorm + Dropout(0.2)

Dense(64, ReLU) + L2(0.0004) + Dropout(0.15)

Dense(32, ReLU) + L2(0.0003) + Dropout(0.1)

Dense(16, ReLU) + L2(0.0003)

Output: Dense(1, linear) for regression

Dependencies

Python 3.8+

TensorFlow / Keras

scikit-learn

NumPy, pandas

Training

Use ReduceLROnPlateau and EarlyStopping callbacks.

Train with validation split of 0.2, batch size 32, for up to 250 epochs.

EarlyStopping restores best weights based on validation MAE.

Evaluation

Evaluate model on test set using MAE and MSE.

Track loss and MAE on training and validation sets.

Feature Engineering (Optional)

Polynomial features can be added for capturing interactions between variables.

Standard scaling is applied after polynomial transformation.

Notes & Tips

Adjust dropout rates and L2 regularization to balance underfitting vs. overfitting.

Monitor training and validation curves to detect overfitting.

Tune learning rate and number of neurons/layers for better performance.
