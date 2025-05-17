import numpy as np
import matplotlib.pyplot as plt
import os
import wfdb
import pywt
import tempfile
import shutil
import urllib.request
import random

# Constants
window_size = 180
classes = ['N', 'L', 'R', 'A', 'V']
class_map = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}
reverse_class_map = {0: 'N', 1: 'L', 2: 'R', 3: 'A', 4: 'V'}

def denoise(data):
    """Clean the signal using wavelet denoising."""
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
    temp_dir = tempfile.mkdtemp()
    base_url = "https://physionet.org/files/mitdb/1.0.0/"
    file_extensions = ['.atr', '.dat', '.hea']
    
    try:
        for ext in file_extensions:
            url = f"{base_url}{record_num}{ext}"
            local_file = os.path.join(temp_dir, f"{record_num}{ext}")
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, local_file)
        
        record = wfdb.rdrecord(os.path.join(temp_dir, record_num), channels=[0])
        annotation = wfdb.rdann(os.path.join(temp_dir, record_num), 'atr')
        return record, annotation, temp_dir
    except Exception as e:
        print(f"Error downloading record {record_num}: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None, None, None

def extract_beat_examples():
    """Extract and visualize example beats from each class."""
    # Records known to contain examples of different classes
    record_map = {
        'N': '100',  # Normal beats
        'L': '109',  # Left bundle branch block
        'R': '124',  # Right bundle branch block
        'A': '209',  # Atrial premature
        'V': '208'   # Ventricular premature
    }
    
    # Try to load from processed data first
    processed_data_dir = os.path.join(os.getcwd(), 'processed_data')
    x_processed_path = os.path.join(processed_data_dir, 'X_processed.npy')
    y_processed_path = os.path.join(processed_data_dir, 'y_processed.npy')
    
    examples = {class_name: [] for class_name in classes}
    
    # If processed data exists, load examples from there
    if os.path.exists(x_processed_path) and os.path.exists(y_processed_path):
        print("Loading examples from processed data...")
        X = np.load(x_processed_path)
        y = np.load(y_processed_path)
        
        # Get examples for each class
        for class_idx, class_name in enumerate(classes):
            class_indices = np.where(y == class_idx)[0]
            if len(class_indices) > 0:
                # Get up to 5 random examples
                sample_indices = np.random.choice(class_indices, min(5, len(class_indices)), replace=False)
                examples[class_name] = [X[i].flatten() for i in sample_indices]
                print(f"Found {len(examples[class_name])} examples of class {class_name}")
            else:
                print(f"No examples found for class {class_name} in processed data")
    
    # If we don't have examples for all classes, download specific records
    missing_classes = [c for c in classes if len(examples[c]) == 0]
    if missing_classes:
        print(f"Downloading examples for missing classes: {missing_classes}")
        for class_name in missing_classes:
            record_num = record_map[class_name]
            record, annotation, temp_dir = download_record(record_num)
            
            if record is None:
                continue
                
            signals = record.p_signal
            r_peaks = annotation.sample
            beat_types = annotation.symbol
            
            # Find beats of the desired class
            class_beats = []
            for i, (r_peak, beat_type) in enumerate(zip(r_peaks, beat_types)):
                if beat_type == class_name:
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
                    
                    class_beats.append(beat_denoised)
                    
                    # Stop after finding 5 examples
                    if len(class_beats) >= 5:
                        break
            
            examples[class_name] = class_beats
            print(f"Downloaded {len(class_beats)} examples of class {class_name}")
            
            # Clean up
            if temp_dir:
                shutil.rmtree(temp_dir)
    
    # Visualize examples
    plt.figure(figsize=(15, 12))
    
    for i, class_name in enumerate(classes):
        class_examples = examples[class_name]
        
        if not class_examples:
            print(f"No examples found for class {class_name}")
            continue
        
        # Plot up to 3 examples for each class
        for j, beat in enumerate(class_examples[:3]):
            plt.subplot(len(classes), 3, i*3 + j + 1)
            plt.plot(beat)
            plt.title(f"Class {class_name}: {get_class_description(class_name)}")
            plt.grid(True)
            
            # Print beat characteristics
            print(f"\nClass {class_name} example {j+1}:")
            print(f"  Mean value: {np.mean(beat):.4f}")
            print(f"  Standard deviation: {np.std(beat):.4f}")
            print(f"  Min value: {np.min(beat):.4f}")
            print(f"  Max value: {np.max(beat):.4f}")
            print(f"  Range: {np.max(beat) - np.min(beat):.4f}")
    
    plt.tight_layout()
    plt.savefig('ecg_class_examples.png')
    plt.show()
    
    return examples

def get_class_description(class_name):
    """Return a description for each class."""
    descriptions = {
        'N': 'Normal Beat',
        'L': 'Left Bundle Branch Block',
        'R': 'Right Bundle Branch Block',
        'A': 'Atrial Premature Contraction',
        'V': 'Ventricular Premature Contraction'
    }
    return descriptions.get(class_name, "Unknown")

def explain_classes():
    """Print detailed explanation of each class."""
    class_explanations = {
        'N': """
        Normal Beat (N):
        - Regular rhythm originating from the sinus node
        - Normal P wave followed by QRS complex and T wave
        - PR interval typically 0.12-0.20 seconds
        - QRS duration typically less than 0.12 seconds
        - Found in healthy individuals and those without conduction abnormalities
        """,
        
        'L': """
        Left Bundle Branch Block (L):
        - QRS complex is widened (>0.12 seconds)
        - Broad, notched R wave in leads I, aVL, V5, V6
        - Absence of Q waves in leads I, V5, V6
        - ST and T wave deflections opposite to the major QRS deflection
        - Indicates a delay or block in conduction through the left bundle branch
        """,
        
        'R': """
        Right Bundle Branch Block (R):
        - QRS complex is widened (>0.12 seconds)
        - RSR' pattern in V1-V3 ("rabbit ears" or M-shaped QRS)
        - Wide S waves in leads I, aVL, V5, V6
        - ST and T wave deflections opposite to terminal QRS deflection
        - Indicates a delay or block in conduction through the right bundle branch
        """,
        
        'A': """
        Atrial Premature Contraction (A):
        - Early beat originating from an ectopic focus in the atria
        - P wave morphology differs from sinus P waves
        - QRS complex usually normal (unless conducted abnormally)
        - Often followed by a compensatory pause
        - Can be caused by caffeine, stress, or underlying heart disease
        """,
        
        'V': """
        Ventricular Premature Contraction (V):
        - Early beat originating from an ectopic focus in the ventricles
        - Wide, bizarre QRS complex (>0.12 seconds)
        - No preceding P wave
        - T wave usually opposite in direction to the QRS
        - Often followed by a full compensatory pause
        - Can occur in healthy individuals or indicate heart disease
        """
    }
    
    print("ECG Beat Classification - Class Explanations:")
    print("=============================================")
    
    for class_name in classes:
        print(class_explanations[class_name])
        print("---------------------------------------------")

if __name__ == "__main__":
    print("ECG Beat Classification - Class Examples")
    print("=======================================")
    
    # Explain the classes
    explain_classes()
    
    # Extract and visualize example beats
    examples = extract_beat_examples()
    
    # Print summary
    print("\nSummary of Examples:")
    for class_name in classes:
        print(f"Class {class_name} ({get_class_description(class_name)}): {len(examples[class_name])} examples")