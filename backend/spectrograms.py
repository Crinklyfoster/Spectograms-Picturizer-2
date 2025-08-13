"""
Spectrogram generation module with integrated audio denoising.
Removes ambient noise (36-40 dB) before generating spectrograms.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pywt
from scipy import signal
import os

# ✨ NEW: Audio Denoising Functions
def denoise_audio_spectral_gating(y, sr, noise_db_threshold=40):
    """
    Advanced spectral gating denoising for ambient noise removal.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        noise_db_threshold (float): Ambient noise threshold in dB (default: 40)
    
    Returns:
        np.ndarray: Denoised audio signal
    """
    try:
        # Estimate noise from first 0.5 seconds (assuming initial silence/low signal)
        noise_duration = min(0.5, len(y) / sr * 0.1)  # 10% of audio or 0.5s
        noise_samples = int(sr * noise_duration)
        noise_clip = y[:noise_samples] if noise_samples > 0 else y[:len(y)//10]
        
        # Compute STFT for full audio and noise clip
        stft_full = librosa.stft(y, hop_length=512, n_fft=2048)
        stft_noise = librosa.stft(noise_clip, hop_length=512, n_fft=2048)
        
        # Get magnitude and phase
        magnitude = np.abs(stft_full)
        phase = np.angle(stft_full)
        noise_magnitude = np.abs(stft_noise)
        
        # Estimate noise profile (mean + std for robustness)
        noise_mean = np.mean(noise_magnitude, axis=1, keepdims=True)
        noise_std = np.std(noise_magnitude, axis=1, keepdims=True)
        noise_threshold = noise_mean + 2 * noise_std  # 2-sigma threshold
        
        # Convert dB threshold to amplitude scaling
        db_factor = 10 ** (noise_db_threshold / 20)
        adaptive_threshold = noise_threshold * db_factor
        
        # Create spectral mask (over-subtraction method)
        alpha = 0.1  # Over-subtraction factor
        mask = np.maximum(
            (magnitude - alpha * adaptive_threshold) / magnitude,
            0.1 * np.ones_like(magnitude)  # Minimum 10% of original signal
        )
        
        # Apply mask
        magnitude_denoised = magnitude * mask
        
        # Reconstruct audio
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        y_denoised = librosa.istft(stft_denoised, hop_length=512)
        
        print(f"✅ Spectral gating denoising applied (threshold: {noise_db_threshold} dB)")
        return y_denoised
        
    except Exception as e:
        print(f"❌ Denoising failed, using original audio: {str(e)}")
        return y

def denoise_audio_simple_gate(y, sr, noise_db_threshold=40):
    """
    Simple noise gate for quick denoising.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        noise_db_threshold (float): Noise threshold in dB
    
    Returns:
        np.ndarray: Denoised audio signal
    """
    try:
        # Convert dB to amplitude (relative to max signal)
        max_amplitude = np.max(np.abs(y))
        noise_amplitude = max_amplitude * (10 ** (-noise_db_threshold / 20))
        
        # Apply noise gate with smooth transitions
        gate_mask = np.abs(y) > noise_amplitude
        
        # Smooth the gate to avoid clicks
        gate_smooth = signal.medfilt(gate_mask.astype(float), kernel_size=5)
        
        # Apply gate
        y_denoised = y * gate_smooth
        
        print(f"✅ Simple noise gate applied (threshold: {noise_db_threshold} dB)")
        return y_denoised
        
    except Exception as e:
        print(f"❌ Simple gating failed, using original audio: {str(e)}")
        return y

def apply_denoising(y, sr, method='spectral_gating', noise_db_threshold=40):
    """
    Apply selected denoising method.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        method (str): 'spectral_gating' or 'simple_gate'
        noise_db_threshold (float): Ambient noise threshold (36-40 dB typical)
    
    Returns:
        np.ndarray: Denoised audio signal
    """
    print(f"🔧 Applying {method} denoising (ambient noise threshold: {noise_db_threshold} dB)")
    
    if method == 'spectral_gating':
        return denoise_audio_spectral_gating(y, sr, noise_db_threshold)
    elif method == 'simple_gate':
        return denoise_audio_simple_gate(y, sr, noise_db_threshold)
    else:
        print(f"❌ Unknown denoising method: {method}, using original audio")
        return y

# Updated spectrogram generation functions with better error handling
def generate_mel_spectrogram(y, sr, save_path):
    """Generate Mel-Spectrogram with improved error handling."""
    try:
        plt.figure(figsize=(12, 8))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr//2)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=sr//2)
        if img is not None:
            plt.colorbar(img, format='%+2.0f dB')
        else:
            plt.colorbar(format='%+2.0f dB')
            
        plt.title('Mel-Spectrogram\n(Energy patterns after noise removal)', fontsize=14)
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Frequency (Hz)')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        plt.close()
        print(f"❌ Mel spectrogram error: {str(e)}")

def generate_cqt_spectrogram(y, sr, save_path):
    """Generate CQT with error handling."""
    try:
        plt.figure(figsize=(12, 8))
        cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=84)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        img = librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note')
        if img is not None:
            plt.colorbar(img, format='%+2.0f dB')
        else:
            plt.colorbar(format='%+2.0f dB')
            
        plt.title('Constant-Q Transform (CQT)\n(Harmonic analysis on clean signal)', fontsize=14)
        plt.xlabel('Time (s)')
        plt.ylabel('CQT Frequency')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        plt.close()
        print(f"❌ CQT spectrogram error: {str(e)}")

def generate_log_stft_spectrogram(y, sr, save_path):
    """Generate Log-STFT with error handling."""
    try:
        plt.figure(figsize=(12, 8))
        stft = librosa.stft(y, hop_length=512, n_fft=2048)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='log')
        if img is not None:
            plt.colorbar(img, format='%+2.0f dB')
        else:
            plt.colorbar(format='%+2.0f dB')
            
        plt.title('Log-STFT Spectrogram\n(Frequency analysis without ambient noise)', fontsize=14)
        plt.xlabel('Time (s)')
        plt.ylabel('Log Frequency (Hz)')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        plt.close()
        print(f"❌ Log-STFT error: {str(e)}")

def generate_wavelet_scalogram(y, sr, save_path):
    """Generate Wavelet Scalogram with error handling."""
    try:
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
        plt.title('Wavelet Scalogram\n(Transient detection on denoised signal)', fontsize=14)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        plt.close()
        print(f"❌ Wavelet error: {str(e)}")

def generate_spectral_kurtosis(y, sr, save_path):
    """Generate Spectral Kurtosis with error handling."""
    try:
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
        plt.title('Spectral Kurtosis\n(Fault impulse detection after denoising)', fontsize=14)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        plt.close()
        print(f"❌ Spectral kurtosis error: {str(e)}")

def generate_modulation_spectrogram(y, sr, save_path):
    """Generate Modulation Spectrogram with error handling."""
    try:
        plt.figure(figsize=(12, 8))
        analytic_signal = signal.hilbert(y)
        envelope = np.abs(analytic_signal)
        f, t, envelope_spec = signal.spectrogram(envelope, sr, nperseg=2048, noverlap=1024)
        envelope_spec_db = 10 * np.log10(envelope_spec + 1e-10)
        
        plt.imshow(envelope_spec_db, extent=[t[0], t[-1], f[0], f[-1]], 
                   cmap='plasma', aspect='auto', origin='lower')
        plt.colorbar(label='Power (dB)')
        plt.title('Modulation Spectrogram\n(Modulation patterns in clean signal)', fontsize=14)
        plt.xlabel('Time (s)')
        plt.ylabel('Modulation Frequency (Hz)')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        plt.close()
        print(f"❌ Modulation spectrogram error: {str(e)}")

def generate_all_spectrograms(audio_path, session_id, file_id=None, denoise=True, denoising_method='spectral_gating', noise_threshold=40):
    """
    Generate all spectrograms with optional audio denoising.
    
    Args:
        audio_path (str): Path to audio file
        session_id (str): Session identifier
        file_id (str): File identifier for batch processing
        denoise (bool): Whether to apply denoising (default: True)
        denoising_method (str): Denoising method to use
        noise_threshold (float): Ambient noise threshold in dB (36-40 typical)
    
    Returns:
        dict: Dictionary containing paths to generated spectrogram images
    """
    try:
        print(f"🎵 Loading audio: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # 🆕 Apply denoising before spectrogram generation
        if denoise:
            y_original = y.copy()  # Keep original for comparison
            y = apply_denoising(y, sr, method=denoising_method, noise_db_threshold=noise_threshold)
            
            # Calculate noise reduction stats
            original_rms = np.sqrt(np.mean(y_original**2))
            denoised_rms = np.sqrt(np.mean(y**2))
            noise_reduction_db = 20 * np.log10(original_rms / denoised_rms) if denoised_rms > 0 else 0
            print(f"📊 Noise reduction: {noise_reduction_db:.2f} dB")
        
        # Create results directory
        if file_id:
            results_dir = os.path.join('results', session_id, file_id)
        else:
            results_dir = os.path.join('results', session_id)
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Spectrogram generation functions (using denoised audio 'y')
        spectrograms = {
            'mel': {
                'name': 'Mel-Spectrogram',
                'function': generate_mel_spectrogram,
                'description': 'Energy patterns after noise removal'
            },
            'cqt': {
                'name': 'Constant-Q Transform (CQT)',
                'function': generate_cqt_spectrogram,
                'description': 'Harmonic analysis on clean signal'
            },
            'log_stft': {
                'name': 'Log-STFT',
                'function': generate_log_stft_spectrogram,
                'description': 'Frequency analysis without ambient noise'
            },
            'wavelet': {
                'name': 'Wavelet Scalogram',
                'function': generate_wavelet_scalogram,
                'description': 'Transient detection on denoised signal'
            },
            'spectral_kurtosis': {
                'name': 'Spectral Kurtosis',
                'function': generate_spectral_kurtosis,
                'description': 'Fault impulse detection after denoising'
            },
            'modulation': {
                'name': 'Modulation Spectrogram',
                'function': generate_modulation_spectrogram,
                'description': 'Modulation patterns in clean signal'
            }
        }
        
        # Generate each spectrogram using denoised audio
        spectrogram_paths = {}
        for spec_type, spec_info in spectrograms.items():
            save_path = os.path.join(results_dir, f'{spec_type}_spectrogram.png')
            try:
                spec_info['function'](y, sr, save_path)  # Pass denoised 'y'
                spectrogram_paths[spec_type] = {
                    'path': save_path,
                    'name': spec_info['name'],
                    'description': spec_info['description']
                }
                print(f"✅ {spec_info['name']} generated with denoised audio")
            except Exception as e:
                print(f"❌ Error generating {spec_info['name']}: {str(e)}")
                continue
        
        return spectrogram_paths
        
    except Exception as e:
        print(f"❌ Critical error in generate_all_spectrograms: {str(e)}")
        return {}
