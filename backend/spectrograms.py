"""
Spectrogram generation module for motor fault detection - Phase 2.
Generates six different types of spectrograms with organized file structure.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pywt
from scipy import signal
import os

# ... (keep all the individual spectrogram generation functions the same) ...

def generate_all_spectrograms(audio_path, session_id, file_id=None):
    """
    Generate all six types of spectrograms for the given audio file.
    
    Args:
        audio_path: Path to the audio file
        session_id: Unique session identifier
        file_id: Unique file identifier (for batch processing)
    
    Returns:
        dict: Dictionary containing paths to generated spectrogram images
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create results directory for this file
    if file_id:
        # Batch processing: each file gets its own folder
        results_dir = os.path.join('results', session_id, file_id)
    else:
        # Single file processing: use session directory
        results_dir = os.path.join('results', session_id)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Define spectrogram types and their generator functions
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

# ... (keep all the individual generation functions the same as before) ...
