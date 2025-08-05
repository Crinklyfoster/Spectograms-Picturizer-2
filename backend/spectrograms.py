"""
Spectrogram generation module - Fixed for hybrid processing.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pywt
from scipy import signal
import os

def generate_mel_spectrogram(y, sr, save_path):
    """Generate Mel-Spectrogram."""
    plt.figure(figsize=(12, 8))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr//2)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=sr//2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram\n(Energy imbalance, tonal shifts, soft degradation patterns)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_cqt_spectrogram(y, sr, save_path):
    """Generate CQT."""
    plt.figure(figsize=(12, 8))
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=84)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q Transform (CQT)\n(Harmonic noise, shifted frequency content)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('CQT Frequency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_log_stft_spectrogram(y, sr, save_path):
    """Generate Log-STFT."""
    plt.figure(figsize=(12, 8))
    stft = librosa.stft(y, hop_length=512, n_fft=2048)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-STFT Spectrogram\n(Low-frequency rumble, imbalance, looseness)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Log Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_wavelet_scalogram(y, sr, save_path):
    """Generate Wavelet Scalogram."""
    plt.figure(figsize=(12, 8))
    if len(y) > 50000:
        y_resampled = signal.resample(y, 50000)
    else:
        y_resampled = y
    
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(y_resampled, scales, 'morl', sampling_period=1/sr)
    
    plt.imshow(np.abs(coefficients), extent=[0, len(y_resampled)/sr, frequencies[-1], frequencies[0]], 
               cmap='hot', aspect='auto', interpolation='bilinear')
    plt.colorbar(label='Magnitude')
    plt.title('Wavelet Scalogram\n(Short bursts, transient spikes from loose components)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_spectral_kurtosis(y, sr, save_path):
    """Generate Spectral Kurtosis."""
    plt.figure(figsize=(12, 8))
    f, t, stft = signal.spectrogram(y, sr, nperseg=2048, noverlap=1024)
    stft_magnitude = np.abs(stft)
    spectral_kurtosis = np.zeros_like(stft_magnitude)
    
    for i in range(stft_magnitude.shape[0]):
        freq_data = stft_magnitude[i, :]
        if np.std(freq_data) > 0:
            mean_val = np.mean(freq_data)
            std_val = np.std(freq_data)
            kurtosis_val = np.mean(((freq_data - mean_val) / std_val) ** 4) - 3
            spectral_kurtosis[i, :] = kurtosis_val
    
    plt.imshow(spectral_kurtosis, extent=[t[0], t[-1], f[0], f[-1]], 
               cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Kurtosis')
    plt.title('Spectral Kurtosis\n(Impulses and sudden power shifts)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_modulation_spectrogram(y, sr, save_path):
    """Generate Modulation Spectrogram."""
    plt.figure(figsize=(12, 8))
    analytic_signal = signal.hilbert(y)
    envelope = np.abs(analytic_signal)
    f, t, envelope_spec = signal.spectrogram(envelope, sr, nperseg=2048, noverlap=1024)
    envelope_spec_db = 10 * np.log10(envelope_spec + 1e-10)
    
    plt.imshow(envelope_spec_db, extent=[t[0], t[-1], f[0], f[-1]], 
               cmap='plasma', aspect='auto', origin='lower')
    plt.colorbar(label='Power (dB)')
    plt.title('Modulation Spectrogram\n(Wobble or sideband-type modulation from winding faults)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Modulation Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_all_spectrograms(audio_path, session_id, file_id=None):
    """Generate all spectrograms with proper file handling."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create results directory
    if file_id:
        # Batch processing: each file gets its own folder
        results_dir = os.path.join('results', session_id, file_id)
    else:
        # Single file processing: use session directory
        results_dir = os.path.join('results', session_id)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Define spectrograms
    spectrograms = {
        'mel': {
            'name': 'Mel-Spectrogram',
            'description': 'Energy imbalance, tonal shifts, soft degradation patterns',
            'function': generate_mel_spectrogram
        },
        'cqt': {
            'name': 'Constant-Q Transform (CQT)',
            'description': 'Harmonic noise, shifted frequency content',
            'function': generate_cqt_spectrogram
        },
        'log_stft': {
            'name': 'Log-STFT',
            'description': 'Low-frequency rumble, imbalance, looseness',
            'function': generate_log_stft_spectrogram
        },
        'wavelet': {
            'name': 'Wavelet Scalogram',
            'description': 'Short bursts, transient spikes from loose components',
            'function': generate_wavelet_scalogram
        },
        'spectral_kurtosis': {
            'name': 'Spectral Kurtosis',
            'description': 'Impulses and sudden power shifts',
            'function': generate_spectral_kurtosis
        },
        'modulation': {
            'name': 'Modulation Spectrogram',
            'description': 'Wobble or sideband-type modulation from winding faults',
            'function': generate_modulation_spectrogram
        }
    }
    
    # Generate each spectrogram
    spectrogram_paths = {}
    for spec_type, spec_info in spectrograms.items():
        save_path = os.path.join(results_dir, f'{spec_type}_spectrogram.png')
        try:
            spec_info['function'](y, sr, save_path)
            spectrogram_paths[spec_type] = {
                'path': save_path,
                'name': spec_info['name'],
                'description': spec_info['description']
            }
        except Exception as e:
            print(f"Error generating {spec_type} spectrogram: {str(e)}")
            continue
    
    return spectrogram_paths
