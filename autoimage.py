import os
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import pyarrow.parquet as pq
import json
import sqlite3
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
from io import BytesIO
import base64

# --- Helper Function to Build CNN Model ---
def build_cnn_model(num_filters=32, kernel_size=3, dropout_rate=0.5, learning_rate=0.00001, input_shape=(128, 128, 3), num_classes=10):
    model = Sequential()
    model.add(Conv2D(num_filters, (kernel_size, kernel_size), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(num_filters * 2, (kernel_size, kernel_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(num_filters * 4, (kernel_size, kernel_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Helper Function to Preprocess Images ---
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- Dataset 1: Directory-Based ---
def load_and_preprocess_directory(dataset_dir, target_size=(128, 128)):
    image_paths = []
    labels = []
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(img_path)
                    labels.append(class_name)
    return np.vstack([preprocess_image(path, target_size) for path in image_paths]), np.array(labels)

# --- Dataset 2: CSV Format ---
def load_and_preprocess_csv(csv_file, target_size=(128, 128)):
    df = pd.read_csv(csv_file)
    images = [preprocess_image(row['image_path'], target_size) for _, row in df.iterrows()]
    labels = df['label'].values
    return np.vstack(images), np.array(labels)

# --- Dataset 3: TFRecord Format ---
def _parse_function(proto):
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    image = tf.io.decode_jpeg(parsed_features['image'])
    image = tf.image.resize(image, [128, 128])
    label = parsed_features['label']
    return image, label

def load_and_preprocess_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset([tfrecord_file])
    dataset = dataset.map(_parse_function)
    images, labels = [], []
    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

# --- Dataset 4: SQLite Format ---
def load_and_preprocess_sqlite(db_file, target_size=(128, 128)):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, label FROM images")
    data = cursor.fetchall()
    images = [preprocess_image(row[0], target_size) for row in data]
    labels = np.array([row[1] for row in data])
    conn.close()
    return np.vstack(images), labels

# --- Dataset 5: HDF5 Format ---
def load_and_preprocess_hdf5(hdf5_file, target_size=(128, 128)):
    """
    Loads images and labels from an HDF5 file.
    Assumes the HDF5 file contains datasets 'images' and 'labels'.
    If the images are stored as raw pixel arrays, they will be normalized to [0,1]
    and resized to the target size if needed.
    """
    with h5py.File(hdf5_file, 'r') as f:
        images = np.array(f['images'])
        labels = np.array(f['labels'])
    
    # Convert images to float32 and normalize (assuming original range is 0-255)
    images = images.astype('float32') / 255.0
    
    # If the images are not already of the target size, resize them.
    # images.shape is expected to be (num_samples, height, width, channels)
    if images.shape[1:3] != target_size:
        resized_images = []
        for img in images:
            # Use TensorFlow's image resizing
            resized_img = tf.image.resize(img, target_size).numpy()
            resized_images.append(resized_img)
        images = np.array(resized_images)
    
    return images, labels

# --- Dataset 6: Parquet Format ---
def load_and_preprocess_parquet(parquet_file, target_size=(128, 128)):
    df = pq.read_table(parquet_file).to_pandas()
    images = [preprocess_image(row['image_path'], target_size) for _, row in df.iterrows()]
    labels = df['label'].values
    return np.vstack(images), labels

# --- Dataset 7: Excel Format ---
def load_and_preprocess_excel(excel_file, target_size=(128, 128)):
    df = pd.read_excel(excel_file)
    images = [preprocess_image(row['image_path'], target_size) for _, row in df.iterrows()]
    labels = df['label'].values
    return np.vstack(images), labels

# --- Dataset 8: JSON Format ---
def load_and_preprocess_json(json_file, target_size=(128, 128)):
    with open(json_file, 'r') as f:
        data = json.load(f)
    images = [preprocess_image(item['image_path'], target_size) for item in data]
    labels = [item['label'] for item in data]
    return np.vstack(images), labels

# --- Dataset 9: Image List in a Text File ---
def load_and_preprocess_image_list(image_list_file, target_size=(128, 128)):
    with open(image_list_file, 'r') as f:
        lines = f.readlines()
    images = [preprocess_image(line.split()[0], target_size) for line in lines]
    labels = [line.split()[1].strip() for line in lines]
    return np.vstack(images), labels

# --- Dataset 10: Image Bytes in CSV File ---
def load_and_preprocess_image_bytes_csv(csv_file, target_size=(128, 128)):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    
    for _, row in df.iterrows():
        try:
            # If the image is base64-encoded
            if isinstance(row['image_bytes'], str):
                img_bytes = base64.b64decode(row['image_bytes'])
            else:
                # If the image is already in bytes format
                img_bytes = row['image_bytes']
            
            # Load image from bytes
            img = image.load_img(BytesIO(img_bytes), target_size=target_size)
            img_array = image.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            images.append(img_array)
            labels.append(row['label'])
        except Exception as e:
            print(f"Error processing row {_}: {str(e)}")
            continue
    
    return np.vstack(images), np.array(labels)

# --- Unified Training Function ---
def train_model(dataset_type, dataset_path):
    if dataset_type == "directory":
        X, y = load_and_preprocess_directory(dataset_path)
    elif dataset_type == "csv":
        X, y = load_and_preprocess_csv(dataset_path)
    elif dataset_type == "tfrecord":
        X, y = load_and_preprocess_tfrecord(dataset_path)
    elif dataset_type == "sqlite":
        X, y = load_and_preprocess_sqlite(dataset_path)
    elif dataset_type == "hdf5":
        X, y = load_and_preprocess_hdf5(dataset_path)
    elif dataset_type == "parquet":
        X, y = load_and_preprocess_parquet(dataset_path)
    elif dataset_type == "excel":
        X, y = load_and_preprocess_excel(dataset_path)
    elif dataset_type == "json":
        X, y = load_and_preprocess_json(dataset_path)
    elif dataset_type == "image_list":
        X, y = load_and_preprocess_image_list(dataset_path)
    elif dataset_type == "image_bytes_csv":
        X, y = load_and_preprocess_image_bytes_csv(dataset_path)
    else:
        raise ValueError("Unsupported dataset type.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = build_cnn_model(num_classes=len(np.unique(y_encoded)))
    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=1)
    model_data = {'model': model, 'label_encoder': label_encoder}
    joblib.dump(model_data, 'cnn_model_with_encoder.pkl')
    test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", test_acc[1])
    return model_data

# --- Unified Prediction Function ---
def predict(image_path, model_data):
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    img_array = preprocess_image(image_path)
    pred = model.predict(img_array)
    predicted_class_encoded = np.argmax(pred, axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_class_encoded)
    return predicted_class[0]

# --- Main Function ---
if __name__ == "__main__":
    # Example Usage
    dataset_type = "directory"  # Change this to the dataset type you want to use
    dataset_path = "Animals"  # Change this to the path of your dataset

    # Train the model
    print("Training the model...")
    model_data = train_model(dataset_type, dataset_path)

    # Predict using the trained model
    test_image_path = "Animals\\cats\\0_0997.jpg"  # Change this to the path of your test image
    predicted_class = predict(test_image_path, model_data)
    print(f"Predicted class: {predicted_class}")
    # Predict using the trained model
    test_image_path = "Animals\\cats\\0_0001.jpg"  # Change this to the path of your test image
    predicted_class = predict(test_image_path, model_data)
    print(f"Predicted class: {predicted_class}")