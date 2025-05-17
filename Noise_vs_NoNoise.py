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

def visualize_denoising():
    """Visualize the effect of denoising on ECG signals."""
    # Try to load processed data first
    processed_data_dir = os.path.join(os.getcwd(), 'processed_data')
    x_processed_path = os.path.join(processed_data_dir, 'X_processed.npy')
    y_processed_path = os.path.join(processed_data_dir, 'y_processed.npy')
    
    # If processed data exists, we need to download original data to see the noisy version
    records_to_try = ['100', '104', '108', '200', '210']  # Records known to have some noise
    
    # Store original and denoised beats for each class
    examples = []
    
    # Download records and extract beats
    for record_num in records_to_try:
        record, annotation, temp_dir = download_record(record_num)
        
        if record is None:
            continue
            
        signals = record.p_signal
        r_peaks = annotation.sample
        beat_types = annotation.symbol
        
        # Find beats with visible noise
        for i, (r_peak, beat_type) in enumerate(zip(r_peaks, beat_types)):
            if beat_type in class_map:
                # Extract window around R-peak
                start = max(0, r_peak - window_size // 2)
                end = min(len(signals), r_peak + window_size // 2)
                
                # Skip if window is not complete
                if end - start != window_size:
                    continue
                
                # Extract the heartbeat segment
                original_beat = signals[start:end, 0]
                
                # Calculate noise level (using high-frequency components as a proxy)
                high_freq_energy = np.sum(np.abs(np.diff(np.diff(original_beat))))
                
                # Apply denoising
                denoised_beat = denoise(original_beat)
                
                # Normalize both signals for fair comparison
                original_normalized = (original_beat - np.mean(original_beat)) / np.std(original_beat)
                denoised_normalized = (denoised_beat - np.mean(denoised_beat)) / np.std(denoised_beat)
                
                # Store the beat if it has enough noise to show difference
                examples.append({
                    'class': beat_type,
                    'record': record_num,
                    'original': original_normalized,
                    'denoised': denoised_normalized,
                    'noise_level': high_freq_energy
                })
                
                # Stop after finding enough examples
                if len(examples) >= 15:
                    break
        
        # Clean up
        if temp_dir:
            shutil.rmtree(temp_dir)
            
        # Stop if we have enough examples
        if len(examples) >= 15:
            break
    
    # Sort examples by noise level
    examples.sort(key=lambda x: x['noise_level'], reverse=True)
    
    # Select examples with different noise levels: high, medium, low
    selected_examples = [
        examples[0],  # High noise
        examples[len(examples)//2],  # Medium noise
        examples[-1]  # Low noise
    ]
    
    # Visualize the original and denoised signals
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('ECG Signal Denoising Comparison', fontsize=16)
    
    noise_levels = ['High Noise', 'Medium Noise', 'Low Noise']
    
    for i, example in enumerate(selected_examples):
        # Original signal
        axes[i, 0].plot(example['original'], 'b')
        axes[i, 0].set_title(f'Original Signal - {noise_levels[i]}')
        axes[i, 0].set_ylabel(f"Class {example['class']}\nRecord {example['record']}")
        axes[i, 0].grid(True)
        
        # Denoised signal
        axes[i, 1].plot(example['denoised'], 'g')
        axes[i, 1].set_title(f'Denoised Signal - {noise_levels[i]}')
        axes[i, 1].grid(True)
        
        # Calculate and display noise reduction metrics
        noise_reduction = np.std(example['original'] - example['denoised'])
        signal_correlation = np.corrcoef(example['original'], example['denoised'])[0, 1]
        
        axes[i, 1].annotate(
            f"Noise reduction: {noise_reduction:.4f}\nSignal correlation: {signal_correlation:.4f}",
            xy=(0.5, 0.05), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            ha='center'
        )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('ecg_denoising_comparison.png')
    plt.show()
    
    # Show detailed analysis of denoising
    visualize_wavelet_decomposition(selected_examples[0]['original'])
    
    return selected_examples

def visualize_wavelet_decomposition(signal):
    """Visualize the wavelet decomposition to understand the denoising process."""
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(signal), w.dec_len)
    
    # Decompose the signal
    coeffs = pywt.wavedec(signal, 'sym4', level=maxlev)
    
    # Create a copy for thresholded coefficients
    thresholded_coeffs = coeffs.copy()
    
    # Apply thresholding to detail coefficients
    for i in range(1, len(coeffs)):
        thresholded_coeffs[i] = pywt.threshold(coeffs[i], 0.04*max(coeffs[i]))
    
    # Reconstruct signal from original coefficients
    original_rec = pywt.waverec(coeffs, 'sym4')
    
    # Reconstruct signal from thresholded coefficients
    denoised_rec = pywt.waverec(thresholded_coeffs, 'sym4')
    
    # Plot the decomposition
    plt.figure(figsize=(15, 12))
    plt.suptitle('Wavelet Decomposition for ECG Denoising', fontsize=16)
    
    # Plot original and reconstructed signals
    plt.subplot(maxlev + 2, 1, 1)
    plt.plot(signal, 'b')
    plt.title('Original Signal')
    plt.grid(True)
    
    # Plot approximation coefficients
    plt.subplot(maxlev + 2, 1, 2)
    plt.plot(coeffs[0], 'r')
    plt.title('Approximation Coefficients (Preserved)')
    plt.grid(True)
    
    # Plot detail coefficients
    for i in range(1, len(coeffs)):
        plt.subplot(maxlev + 2, 1, i + 2)
        plt.plot(coeffs[i], 'g', label='Original')
        plt.plot(thresholded_coeffs[i], 'r', label='Thresholded')
        plt.title(f'Detail Coefficients Level {i}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('ecg_wavelet_decomposition.png')
    plt.show()
    
    # Plot original vs denoised
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal, 'b', label='Original')
    plt.plot(denoised_rec, 'r', label='Denoised')
    plt.title('Original vs Denoised Signal')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(signal - denoised_rec, 'g')
    plt.title('Removed Noise Component')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ecg_denoising_result.png')
    plt.show()

def add_fake_noise(examples):
    """Add synthetic noise to clean signals to better demonstrate denoising."""
    # Types of synthetic noise to demonstrate
    noise_types = {
        'Gaussian': lambda x: x + np.random.normal(0, 0.2, len(x)),
        'Power Line (60Hz)': lambda x: x + 0.2 * np.sin(2 * np.pi * 60 * np.linspace(0, 1, len(x))),
        'Baseline Wander': lambda x: x + 0.5 * np.sin(2 * np.pi * 0.5 * np.linspace(0, 1, len(x)))
    }
    
    # Select a clean example
    clean_example = examples[-1]['denoised']  # Use the cleanest example's denoised signal
    
    plt.figure(figsize=(15, 12))
    plt.suptitle('Denoising Different Types of ECG Noise', fontsize=16)
    
    # For each noise type
    for i, (noise_name, noise_func) in enumerate(noise_types.items()):
        # Add synthetic noise
        noisy_signal = noise_func(clean_example.copy())
        
        # Denoise
        denoised_signal = denoise(noisy_signal)
        
        # Plot
        plt.subplot(3, 2, i*2 + 1)
        plt.plot(noisy_signal, 'b')
        plt.title(f'Signal with {noise_name} Noise')
        plt.grid(True)
        
        plt.subplot(3, 2, i*2 + 2)
        plt.plot(denoised_signal, 'g')
        plt.title(f'After Denoising')
        plt.grid(True)
        
        # Calculate and display noise reduction metrics
        noise_reduction = np.std(noisy_signal - denoised_signal)
        signal_correlation = np.corrcoef(clean_example, denoised_signal)[0, 1]
        
        plt.annotate(
            f"Noise reduction: {noise_reduction:.4f}\nSignal correlation: {signal_correlation:.4f}",
            xy=(0.5, 0.05), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            ha='center'
        )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('synthetic_noise_denoising.png')
    plt.show()

if __name__ == "__main__":
    print("ECG Signal Denoising Visualization")
    print("==================================")
    
    # Visualize denoising on real ECG signals
    examples = visualize_denoising()
    
    # Demonstrate denoising with synthetic noise
    add_fake_noise(examples)
    
 
# Explanation of the Denoising Process:

# 1. The wavelet transform decomposes the signal into different frequency components
# 2. The high-frequency noise is primarily contained in the detail coefficients
# 3. Thresholding is applied to remove small coefficients which likely represent noise
# 4. The signal is reconstructed from the thresholded coefficients
# 5. This preserves the important ECG features while removing noise
# 

# 
# Types of Noise in ECG Signals:

# 1. Baseline Wander: Low-frequency noise caused by patient breathing or movement
# 2. Power Line Interference: 50/60 Hz noise from electrical power systems
# 3. Muscle Artifacts: High-frequency noise from muscle activity
# 4. Motion Artifacts: Irregular noise caused by electrode or patient movement
# 5. Instrument Noise: Noise from the ECG recording device itself
#
    