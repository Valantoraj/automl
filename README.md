# AutoML Suite

This repository contains three Python scripts for automating machine learning tasks, chatbot development, and image classification. Each script is designed to handle specific tasks with minimal setup and configuration.

## Features

### 1. **AutoML (automl.py)**
- **Purpose**: Automates the machine learning pipeline for classification and regression tasks.
- **Supported Models**: Logistic Regression, Random Forest, SVM, Decision Tree, KNN, Gradient Boosting, AdaBoost, LightGBM, CatBoost, Ridge.
- **Preprocessing**: Handles numerical, categorical, and text data with imputation, scaling, and encoding.
- **Hyperparameter Tuning**: Uses GridSearchCV for optimizing model parameters.
- **Model Saving**: Saves the best-performing model for future use.

### 2. **Chatbot (autochat.py)**
- **Purpose**: Implements a chatbot using LSTM or Transformer-based models.
- **Dynamic Tuning**: Automatically tunes hyperparameters based on dataset statistics.
- **Interactive Chat**: Provides an interactive chat loop for testing the chatbot.
- **Intents**: Supports JSON-based intents for defining chatbot behavior.

### 3. **Image Classification (autoimage.py)**
- **Purpose**: Automates image classification using a Convolutional Neural Network (CNN).
- **Supported Dataset Formats**: Directory-based, CSV, TFRecord, SQLite, HDF5, Parquet, Excel, JSON, image list, and image bytes in CSV.
- **Unified Interface**: Provides a single function for training and prediction across all dataset formats.
- **Model Saving**: Saves the trained model and label encoder for future use.

### Running The Code

## automl.py

from automl import train_model

file_path = "path/to/your/dataset.csv"
target_column = "target"
model = train_model(file_path, target_column)

## -----------------

## autochat.py

from autochat import main

file_path = "path/to/your/intents_file"
main(file_path)

## -----------------

## autoimage.py

from autoimage import train_model, predict

dataset_type = "directory"
dataset_path = "path/to/your/dataset"
model_data = train_model(dataset_type, dataset_path)

test_image_path = "path/to/test/image.jpg"
predicted_class = predict(test_image_path, model_data)
print(f"Predicted class: {predicted_class}")