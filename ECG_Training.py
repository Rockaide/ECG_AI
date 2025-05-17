import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pywt
import wfdb
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import itertools
import requests
import tempfile
import random
import shutil
import urllib.request

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Constants - declare these as global variables
window_size = 180
classes = ['N', 'L', 'R', 'A', 'V']
n_classes = len(classes)
class_map = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}

def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04
    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    datarec = pywt.waverec(coeffs, 'sym4')
    return datarec

def download_record(record_num):
    """Download a record from PhysioNet directly."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Base URL for the MIT-BIH Arrhythmia Database
    base_url = "https://physionet.org/files/mitdb/1.0.0/"
    
    # Files to download
    file_extensions = ['.atr', '.dat', '.hea']
    
    try:
        for ext in file_extensions:
            url = f"{base_url}{record_num}{ext}"
            local_file = os.path.join(temp_dir, f"{record_num}{ext}")
            
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, local_file)
        
        # Read the record using wfdb
        record = wfdb.rdrecord(os.path.join(temp_dir, record_num), channels=[0])
        annotation = wfdb.rdann(os.path.join(temp_dir, record_num), 'atr')
        
        return record, annotation, temp_dir
        
    except Exception as e:
        print(f"Error downloading record {record_num}: {e}")
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        return None, None, None

def extract_heartbeats(record_nums=None):
    """Extract heartbeats from the specified records."""
    global n_classes, classes, class_map
    
    if record_nums is None:
        # For full dataset
        record_nums = ['100', '101', '102', '103', '104', '105', '106', '107', 
                       '108', '109', '111', '112', '113', '114', '115', '116', 
                       '117', '118', '119', '121', '122', '123', '124', '200', 
                       '201', '202', '203', '205', '207', '208', '209', '210', 
                       '212', '213', '214', '215', '217', '219', '220', '221', 
                       '222', '223', '228', '230', '231', '232', '233', '234']
    
    # Create a local directory for MIT-BIH data if it doesn't exist
    local_data_dir = os.path.join(os.getcwd(), 'mitdb_data')
    os.makedirs(local_data_dir, exist_ok=True)
    
    X = []
    y = []
    count_classes = [0] * n_classes
    
    for record_num in record_nums:
        print(f"Processing record {record_num}...")
        
        # Check if the record files already exist locally
        record_files_exist = all(
            os.path.exists(os.path.join(local_data_dir, f"{record_num}{ext}"))
            for ext in ['.atr', '.dat', '.hea']
        )
        
        if record_files_exist:
            print(f"Record {record_num} found locally, using cached data.")
            try:
                # Read the local record
                record = wfdb.rdrecord(os.path.join(local_data_dir, record_num), channels=[0])
                annotation = wfdb.rdann(os.path.join(local_data_dir, record_num), 'atr')
                temp_dir = None  # No temp directory needed
            except Exception as e:
                print(f"Error reading local record {record_num}: {e}")
                record_files_exist = False  # Try downloading instead
        
        # If files don't exist locally or couldn't be read, download them
        if not record_files_exist:
            print(f"Record {record_num} not found locally, downloading...")
            try:
                # Download the record to a temporary directory first
                record, annotation, temp_dir = download_record(record_num)
                
                if record is None:
                    continue
                
                # Copy the downloaded files to our local data directory
                for ext in ['.atr', '.dat', '.hea']:
                    src_file = os.path.join(temp_dir, f"{record_num}{ext}")
                    dst_file = os.path.join(local_data_dir, f"{record_num}{ext}")
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, dst_file)
                        print(f"Saved {dst_file} for future use")
                
                # Clean up the temporary directory
                if temp_dir:
                    shutil.rmtree(temp_dir)
                    temp_dir = None
            except Exception as e:
                print(f"Error downloading record {record_num}: {e}")
                if temp_dir:
                    shutil.rmtree(temp_dir)
                continue
        
        try:
            # Get signal data
            signals = record.p_signal
            
            # Get R-peak locations and beat types
            r_peaks = annotation.sample
            beat_types = annotation.symbol
            
            for i, (r_peak, beat_type) in enumerate(zip(r_peaks, beat_types)):
                if beat_type in class_map and count_classes[class_map[beat_type]] < 10000:
                    # Extract window around R-peak
                    start = max(0, r_peak - window_size // 2)
                    end = min(len(signals), r_peak + window_size // 2)
                    
                    # Skip if window is not complete
                    if end - start != window_size:
                        continue
                    
                    # Extract the heartbeat segment
                    beat = signals[start:end, 0]
                    
                    # Apply denoising
                    beat_denoised = denoise(beat)
                    
                    # Normalize
                    beat_denoised = (beat_denoised - np.mean(beat_denoised)) / np.std(beat_denoised)
                    
                    X.append(beat_denoised)
                    y.append(class_map[beat_type])
                    count_classes[class_map[beat_type]] += 1
                    
                    # Print progress occasionally
                    if sum(count_classes) % 100 == 0:
                        print(f"Processed {sum(count_classes)} beats. Current distribution: {count_classes}")
        
        except Exception as e:
            print(f"Error processing record {record_num} data: {e}")
            continue
    
    # Save processed data for future use
    if len(X) > 0:
        processed_data_dir = os.path.join(os.getcwd(), 'processed_data')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Save as numpy arrays
        np.save(os.path.join(processed_data_dir, 'X_processed.npy'), np.array(X))
        np.save(os.path.join(processed_data_dir, 'y_processed.npy'), np.array(y))
        
        print(f"Saved processed data to {processed_data_dir} for future use")
    
    if len(X) == 0:
        raise ValueError("No data was loaded. Check your internet connection and try again.")
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for CNN input
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, count_classes

def load_kaggle_dataset():
    """
    Alternative way to load a pre-processed version of the MIT-BIH dataset from Kaggle.
    Requires downloading the dataset from:
    https://www.kaggle.com/datasets/shayanfazeli/heartbeat
    """
    global n_classes, classes
    
    try:
        # Ask for file paths
        train_path = input("Enter path to mitbih_train.csv (or press Enter to skip): ")
        if not train_path:
            return None, None, None
            
        test_path = input("Enter path to mitbih_test.csv (or press Enter to skip): ")
        if not test_path:
            return None, None, None
            
        # Check if files exist
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("One or both files not found.")
            return None, None, None
            
        # Load data
        print("Loading training data...")
        train_data = pd.read_csv(train_path, header=None)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        print("Loading test data...")
        test_data = pd.read_csv(test_path, header=None)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        # Reshape for CNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Combine train and test for our pipeline
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        
        # Count classes
        count_classes = [np.sum(y == i) for i in range(n_classes)]
        
        print(f"Loaded {len(X)} samples with distribution: {count_classes}")
        
        return X, y, count_classes
        
    except Exception as e:
        print(f"Error loading Kaggle dataset: {e}")
        return None, None, None

def generate_synthetic_data():
    """Generate synthetic data for demonstration purposes."""
    global n_classes, classes
    
    print("Generating synthetic ECG data...")
    
    # Number of examples per class
    n_examples = 1000
    
    X = []
    y = []
    
    for class_idx in range(n_classes):
        # Generate base signal (sine wave with different frequencies for each class)
        t = np.linspace(0, 2*np.pi, window_size)
        frequency = 1 + class_idx * 0.5  # Different frequency for each class
        
        for i in range(n_examples):
            # Base signal
            signal = np.sin(frequency * t)
            
            # Add R-peak (simulated)
            peak_height = 1.0 + 0.2 * class_idx
            peak_width = 10
            peak_pos = window_size // 2
            
            # Create peak
            peak = np.zeros(window_size)
            peak[max(0, peak_pos-peak_width//2):min(window_size, peak_pos+peak_width//2)] = peak_height
            
            # Combine
            combined = signal + peak
            
            # Add noise
            noise = np.random.normal(0, 0.1, window_size)
            final_signal = combined + noise
            
            # Normalize
            final_signal = (final_signal - np.mean(final_signal)) / np.std(final_signal)
            
            X.append(final_signal)
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for CNN
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    count_classes = [n_examples] * n_classes
    
    return X, y, count_classes

def balance_classes(X, y, count_classes):
    """Balance classes by oversampling minority classes."""
    max_samples = max(count_classes)
    X_balanced = []
    y_balanced = []
    
    for i in range(len(count_classes)):
        # Get indices for this class
        indices = np.where(y == i)[0]
        
        if len(indices) == 0:
            continue
            
        # If we need to oversample
        if len(indices) < max_samples:
            # Get original samples
            X_class = X[indices]
            y_class = y[indices]
            
            # Oversample
            X_resampled, y_resampled = resample(X_class, y_class, 
                                               replace=True, 
                                               n_samples=max_samples,
                                               random_state=42)
            
            X_balanced.extend(X_resampled)
            y_balanced.extend(y_resampled)
        else:
            # If we have enough or too many samples, just take max_samples
            X_balanced.extend(X[indices[:max_samples]])
            y_balanced.extend(y[indices[:max_samples]])
    
    return np.array(X_balanced), np.array(y_balanced)

def create_model(input_shape, n_classes):
    """Create a CNN model for ECG classification."""
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                 optimizer=Adam(learning_rate=0.0001),
                 metrics=['accuracy'])
    
    return model

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    # Ensure we can access global variables
    global classes, n_classes
    
    # Data loading strategy
    X, y, count_classes = None, None, None
    
    # Try multiple data loading methods
    data_loading_methods = [
        (extract_heartbeats, "Extracting heartbeats from MIT-BIH records"),
        (load_kaggle_dataset, "Loading pre-processed Kaggle dataset"),
        (generate_synthetic_data, "Generating synthetic data")
    ]
    
    for method, description in data_loading_methods:
        if X is None:
            print(f"\n--- {description} ---")
            try:
                X, y, count_classes = method()
                if X is not None:
                    print(f"Successfully loaded data with {description.lower()}!")
                    break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    if X is None:
        print("Failed to load data using any method. Exiting.")
        return
    
    # Check if we have data for all classes
    if min(count_classes) == 0:
        missing_classes = [classes[i] for i in range(n_classes) if count_classes[i] == 0]
        print(f"Warning: No data for classes: {missing_classes}")
        print("Filtering out classes with no data...")
        
        valid_class_indices = [i for i in range(n_classes) if count_classes[i] > 0]
        valid_classes = [classes[i] for i in valid_class_indices]
        
        # Create a mapping from old class indices to new ones
        class_map_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_class_indices)}
        
        # Filter data to include only valid classes
        valid_indices = [i for i, label in enumerate(y) if label in valid_class_indices]
        X = X[valid_indices]
        y_new = np.array([class_map_new[label] for label in y[valid_indices]])
        y = y_new
        
        # Update class-related variables
        classes = valid_classes
        n_classes = len(classes)
        count_classes = [np.sum(y == i) for i in range(n_classes)]
        
        print(f"Continuing with {n_classes} classes: {classes}")
        print(f"Updated class distribution: {count_classes}")
    
    # Save the raw data
    print("Saving raw data...")
    os.makedirs('data', exist_ok=True)
    np.save('data/X_raw.npy', X)
    np.save('data/y_raw.npy', y)
    
    # Balance classes
    print("Balancing classes...")
    X_balanced, y_balanced = balance_classes(X, y, count_classes)
    balanced_counts = [np.sum(y_balanced == i) for i in range(n_classes)]
    print("Balanced class distribution:", balanced_counts)
    
    # Convert labels to one-hot encoding
    y_categorical = to_categorical(y_balanced, num_classes=n_classes)
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_balanced, y_categorical, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Create and train model
    print("Creating model...")
    model = create_model(input_shape=(window_size, 1), n_classes=n_classes)
    model.summary()
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=30,  # Reduced for faster execution
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]:.4f}')
    print(f'Test accuracy: {score[1]:.4f}')
    
    # Plot training history
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.show()
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=classes))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()
    
    # Save the model
    model.save('models/ecg_classification_model.h5')
    print("Model saved as 'models/ecg_classification_model.h5'")
    
    # Example of using the model for prediction
    print("\nExample prediction:")
    example_idx = np.random.randint(0, len(X_test))
    example_beat = X_test[example_idx]
    true_class = classes[y_true_classes[example_idx]]
    
    # Make prediction
    pred = model.predict(np.expand_dims(example_beat, axis=0))[0]
    pred_class = classes[np.argmax(pred)]
    
    # Plot the beat
    plt.figure(figsize=(12, 4))
    plt.plot(example_beat.flatten())
    plt.title(f'Example ECG Beat - True: {true_class}, Predicted: {pred_class}')
    plt.grid(True)
    plt.savefig('plots/example_prediction.png')
    plt.show()
    
    print(f"True class: {true_class}")
    print(f"Predicted class: {pred_class}")
    print(f"Prediction probabilities: {pred}")

if __name__ == "__main__":
    main()