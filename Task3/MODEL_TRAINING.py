#!/usr/bin/env python3
import os
import glob
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

# Optionally enable GPU memory growth if a GPU is available.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print(e)

# -----------------------------
# Model Definitions (unchanged)
# -----------------------------

def build_era_classifier(input_shape=(224, 224, 3), num_classes=3):
    """
    Build a simple CNN for era classification.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name='era_classifier')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_colorization_model(image_shape=(256, 256, 1), era_vector_length=3):
    """
    Build a simple U-Net–like colorization model that accepts a grayscale image and an era-conditioning vector.
    """
    # Inputs: grayscale image and era-conditioning vector.
    image_input = tf.keras.Input(shape=image_shape, name='image_input')
    era_input   = tf.keras.Input(shape=(era_vector_length,), name='era_input')
    
    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(image_input)
    c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    
    c2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    
    # Bottleneck
    bn = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    
    # Expand era vector to match spatial dimensions.
    bn_shape = tf.keras.backend.int_shape(bn)  # (None, H, W, channels)
    h_bn, w_bn = bn_shape[1], bn_shape[2]
    era_dense = tf.keras.layers.Dense(h_bn * w_bn * 8, activation='relu')(era_input)
    era_reshape = tf.keras.layers.Reshape((h_bn, w_bn, 8))(era_dense)
    bn_concat = tf.keras.layers.Concatenate()([bn, era_reshape])
    
    # Decoder
    u1 = tf.keras.layers.UpSampling2D((2,2))(bn_concat)
    u1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(u1)
    u2 = tf.keras.layers.UpSampling2D((2,2))(u1)
    u2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(u2)
    output = tf.keras.layers.Conv2D(3, (1,1), activation='sigmoid')(u2)
    
    model = tf.keras.Model(inputs=[image_input, era_input], outputs=output, name='colorization_model')
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------------
# Dataset Loader for Upgraded Dataset
# -----------------------------
# These functions use Python’s glob to list files in nested directories.
# The label (era) is determined by the top-level folder name under data_dir.

def list_image_files(data_dir, valid_exts=('.jpg','.jpeg','.png','.gif','.bmp')):
    pattern = os.path.join(data_dir, '*', '*', '*')
    files = glob.glob(pattern)
    # Filter files by extension (case-insensitive)
    files = [f for f in files if f.lower().endswith(valid_exts)]
    return files

def get_era_from_filepath(filepath):
    # Assumes structure: data_dir/ERA/QUERY/filename
    parts = os.path.normpath(filepath).split(os.sep)
    # The era is the folder immediately under the base directory.
    # For example, if base_dir is "dataset/historical_photos_by_era", then era is parts[-3].
    return parts[-3]

def load_era_dataset_upgraded(data_dir, img_size=(224,224), batch_size=16, validation_split=0.2, subset='training', class_names=None):
    """
    Loads an era classification dataset from an upgraded directory structure.
    Returns a tf.data.Dataset and the class_names.
    """
    file_list = list_image_files(data_dir)
    if class_names is None:
        # Automatically determine unique eras from file paths.
        eras = sorted({get_era_from_filepath(f) for f in file_list})
    else:
        eras = class_names
    print("Detected era classes:", eras)
    
    # Create lists for file paths and labels.
    file_paths = []
    labels = []
    for f in file_list:
        era = get_era_from_filepath(f)
        if era in eras:
            file_paths.append(f)
            labels.append(eras.index(era))
    
    # Convert labels to one-hot encoding.
    num_classes = len(eras)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Create a tf.data.Dataset from the file paths and labels.
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def _process(file_path, label):
        # Read image from file path.
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    
    # Shuffle dataset.
    ds = ds.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=False)
    
    # Split dataset into training and validation.
    total = len(file_paths)
    val_size = int(total * validation_split)
    if subset == 'training':
        ds = ds.skip(val_size)
    else:
        ds = ds.take(val_size)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, eras

def load_colorization_dataset_upgraded(data_dir, img_size=(256,256), batch_size=16, validation_split=0.2, subset='training', class_names=None):
    """
    Loads a colorization dataset from an upgraded directory structure.
    For each image, converts the image to grayscale (input) and uses the original as target.
    Also extracts the era label from the file path.
    """
    file_list = list_image_files(data_dir)
    if class_names is None:
        eras = sorted({get_era_from_filepath(f) for f in file_list})
    else:
        eras = class_names
    print("Detected era classes for colorization:", eras)
    
    file_paths = []
    labels = []
    for f in file_list:
        era = get_era_from_filepath(f)
        if era in eras:
            file_paths.append(f)
            labels.append(eras.index(era))
    
    num_classes = len(eras)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def _process(file_path, label):
        # Read and decode image.
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=False)
    
    # For colorization: input is (grayscale image, era label) and target is original color image.
    def process_for_colorization(image, label):
        gray = tf.image.rgb_to_grayscale(image)
        return (gray, label), image
    
    ds = ds.map(process_for_colorization, num_parallel_calls=tf.data.AUTOTUNE)
    
    total = len(file_paths)
    val_size = int(total * validation_split)
    if subset == 'training':
        ds = ds.skip(val_size)
    else:
        ds = ds.take(val_size)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, eras

# -----------------------------
# Training Functions (updated to use upgraded dataset loaders)
# -----------------------------

def train_era_classifier_model(data_dir):
    # Specify class names if desired; otherwise they are determined automatically.
    class_names = ['1900s', '1950s', '1970s']
    train_ds, detected_classes = load_era_dataset_upgraded(
        data_dir, img_size=(224,224), batch_size=16, validation_split=0.2, subset='training', class_names=class_names)
    val_ds, _ = load_era_dataset_upgraded(
        data_dir, img_size=(224,224), batch_size=16, validation_split=0.2, subset='validation', class_names=class_names)
    
    num_classes = len(detected_classes)
    print(f"Era Classification - Using classes: {detected_classes} ({num_classes} classes)")
    
    model = build_era_classifier(input_shape=(224,224,3), num_classes=num_classes)
    model.summary()
    
    print("Training era classifier model...")
    model.fit(train_ds, validation_data=val_ds, epochs=5)
    model.save('era_classifier.h5')
    print("Era classifier model saved as era_classifier.h5")

def train_colorization_model(data_dir):
    class_names = ['1900s', '1950s', '1970s']
    train_ds, detected_classes = load_colorization_dataset_upgraded(
        data_dir, img_size=(256,256), batch_size=16, validation_split=0.2, subset='training', class_names=class_names)
    val_ds, _ = load_colorization_dataset_upgraded(
        data_dir, img_size=(256,256), batch_size=16, validation_split=0.2, subset='validation', class_names=class_names)
    
    num_classes = len(detected_classes)
    print(f"Colorization - Using era vector length: {num_classes}")
    
    model = build_colorization_model(image_shape=(256,256,1), era_vector_length=num_classes)
    model.summary()
    
    print("Training colorization model...")
    model.fit(train_ds, validation_data=val_ds, epochs=5)
    model.save('colorization_model.h5')
    print("Colorization model saved as colorization_model.h5")

# -----------------------------
# Main Execution
# -----------------------------

def main():
    # Directory where your upgraded dataset is stored.
    data_dir = 'dataset/historical_photos_by_era'
    
    print("Starting training of the Era Classifier Model with upgraded dataset...")
    train_era_classifier_model(data_dir)
    
    print("\nStarting training of the Colorization Model with upgraded dataset...")
    train_colorization_model(data_dir)

if __name__ == '__main__':
    main()
