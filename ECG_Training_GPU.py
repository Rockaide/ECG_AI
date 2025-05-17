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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import regularizers
import itertools
import requests
import tempfile
import random
import shutil
import urllib.request
import time
import datetime

# GPU Configuration
def configure_gpu():
    """Configure TensorFlow to use GPU efficiently."""
    print("Checking for GPU availability...")
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            
            # Set TensorFlow to use the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            print("GPU configured successfully!")
            return True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU found. Training will use CPU.")
        return False

# Call this at the beginning of your script
has_gpu = configure_gpu()

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

def load_or_extract_heartbeats(record_nums=None):
    """Load processed data if available, otherwise extract heartbeats from records."""
    global n_classes
    
    # Check if processed data already exists
    processed_data_dir = os.path.join(os.getcwd(), 'processed_data')
    x_processed_path = os.path.join(processed_data_dir, 'X_processed.npy')
    y_processed_path = os.path.join(processed_data_dir, 'y_processed.npy')
    
    if os.path.exists(x_processed_path) and os.path.exists(y_processed_path):
        print("Found pre-processed data, loading...")
        try:
            X = np.load(x_processed_path)
            y = np.load(y_processed_path)
            
            # Calculate class distribution
            count_classes = [np.sum(y == i) for i in range(n_classes)]
            print(f"Loaded processed data with distribution: {count_classes}")
            
            return X, y, count_classes
        except Exception as e:
            print(f"Error loading processed data: {e}")
            print("Will extract heartbeats from original records instead.")
    
    # If we don't have processed data or couldn't load it, extract from records
    return extract_heartbeats(record_nums)

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

def augment_beat(beat, amplitude_range=(0.8, 1.2), noise_range=(0.01, 0.05)):
    """Apply random augmentation to an ECG beat."""
    # Random amplitude scaling
    amplitude_scale = np.random.uniform(amplitude_range[0], amplitude_range[1])
    augmented_beat = beat * amplitude_scale
    
    # Add random noise
    noise_level = np.random.uniform(noise_range[0], noise_range[1])
    noise = np.random.normal(0, noise_level, beat.shape)
    augmented_beat += noise
    
    # Random time shift (small)
    shift = np.random.randint(-5, 5)  # Shift by up to 5 samples
    if shift != 0:
        augmented_beat = np.roll(augmented_beat, shift, axis=0)
    
    # Normalize again after augmentation
    augmented_beat = (augmented_beat - np.mean(augmented_beat)) / np.std(augmented_beat)
    
    return augmented_beat

class ECGDataGenerator(tf.keras.utils.Sequence):
    """Generator to efficiently feed data to GPU during training."""
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.X))
        self.on_epoch_end()
        
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X_batch = self.X[indexes]
        y_batch = self.y[indexes]
        
        # Apply augmentation if enabled
        if self.augment:
            X_batch = np.array([augment_beat(beat) for beat in X_batch])
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

def balance_classes(X, y, count_classes):
    """Balance classes by augmenting data for minority classes."""
    max_samples = max(count_classes)
    X_balanced = []
    y_balanced = []
    
    for i in range(len(count_classes)):
        # Get indices for this class
        indices = np.where(y == i)[0]
        
        if len(indices) == 0:
            continue
        
        # Get samples for this class
        X_class = X[indices]
        
        # If we need to augment (for minority classes)
        if len(indices) < max_samples:
            # Original samples
            X_balanced.extend(X_class)
            y_balanced.extend([i] * len(X_class))
            
            # Number of augmented samples needed
            n_augment = max_samples - len(indices)
            
            # Generate augmented samples
            for _ in range(n_augment):
                # Randomly select a sample to augment
                idx = np.random.randint(0, len(X_class))
                # Augment the selected beat
                augmented_beat = augment_beat(X_class[idx])
                X_balanced.append(augmented_beat)
                y_balanced.append(i)
        else:
            # For majority classes, take a random subset
            selected_indices = np.random.choice(indices, max_samples, replace=False)
            X_balanced.extend(X[selected_indices])
            y_balanced.extend([i] * max_samples)
    
    return np.array(X_balanced), np.array(y_balanced)

def create_model(input_shape, n_classes):
    """Create a CNN model for ECG classification with GPU optimizations."""
    model = Sequential()
    
    # Optimized for GPU with power-of-2 filter sizes
    model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape, 
                    padding='same', kernel_regularizer=regularizers.l2(0.003)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(64, kernel_size=5, activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(0.003)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(128, kernel_size=5, activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(0.003)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.003)))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    
    # Use a higher learning rate with GPU
    lr = 0.001 if has_gpu else 0.0001
    opt = Adam(learning_rate=lr)
    
    # Use mixed precision for GPU
    if has_gpu:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                optimizer=opt,
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

def benchmark_training():
    """Benchmark training speed on CPU vs GPU."""
    global window_size, n_classes
    
    # Create small synthetic dataset for benchmarking
    X_bench = np.random.randn(1000, window_size, 1)
    y_bench = to_categorical(np.random.randint(0, n_classes, 1000), n_classes)
    
    # Create model
    model = create_model(input_shape=(window_size, 1), n_classes=n_classes)
    
    # Force CPU
    print("Benchmarking on CPU...")
    with tf.device('/CPU:0'):
        start_time = time.time()
        model.fit(X_bench, y_bench, batch_size=32, epochs=5, verbose=0)
        cpu_time = time.time() - start_time
    
    # Reset model
    model = create_model(input_shape=(window_size, 1), n_classes=n_classes)
    
    # Try GPU if available
    if has_gpu:
        print("Benchmarking on GPU...")
        with tf.device('/GPU:0'):
            start_time = time.time()
            model.fit(X_bench, y_bench, batch_size=32, epochs=5, verbose=0)
            gpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time:.2f}s, GPU time: {gpu_time:.2f}s")
        print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print(f"CPU time: {cpu_time:.2f}s (No GPU available)")

def main():
    # Ensure we can access global variables
    global classes, n_classes
    
    # Optional: Run benchmark to compare CPU vs GPU speed
    if has_gpu:
        benchmark_training()
    
    # Enable mixed precision for faster GPU training
    if has_gpu:
        print("Enabling mixed precision training for faster GPU performance...")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    # Data loading strategy
    X, y, count_classes = None, None, None
    
    # Try multiple data loading methods
    data_loading_methods = [
        (load_or_extract_heartbeats, "Loading or extracting heartbeats"),
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
    
    # Split data first before balancing to prevent data leakage
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Balance only the training data
    print("Balancing training data...")
    count_classes_train = [np.sum(y_train_raw == i) for i in range(n_classes)]
    X_train_balanced, y_train_balanced = balance_classes(X_train_raw, y_train_raw, count_classes_train)
    
    # Split training into actual training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_balanced, y_train_balanced, test_size=0.2, random_state=42
    )
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_val_cat = to_categorical(y_val, num_classes=n_classes)
    y_test_cat = to_categorical(y_test, num_classes=n_classes)
    
    # Print dataset sizes
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Create model
    print("Creating model...")
    model = create_model(input_shape=(window_size, 1), n_classes=n_classes)
    model.summary()
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint(
        'models/best_model.h5', 
        monitor='val_loss',
        save_best_only=True, 
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Add TensorBoard callback
    log_dir = os.path.join(os.getcwd(), "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Then create the TensorBoard callback
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # Optimize batch size for GPU
    batch_size = 64 if has_gpu else 32
    
    # Create data generators for efficient training
    train_generator = ECGDataGenerator(
        X_train, y_train_cat, 
        batch_size=batch_size, 
        shuffle=True,
        augment=True  # Enable augmentation
    )
    
    val_generator = ECGDataGenerator(
        X_val, y_val_cat,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    
    # Modified fit call without multiprocessing arguments
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    score = model.evaluate(X_test, y_test_cat, verbose=1)
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
    y_true_classes = np.argmax(y_test_cat, axis=1)
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=classes))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()
    
    # Generate normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=classes, normalize=True, title='Normalized Confusion Matrix')
    plt.savefig('plots/normalized_confusion_matrix.png')
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