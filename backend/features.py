"""
Feature extraction module for motor fault detection.
Extracts time-domain and frequency-domain features from audio signals.
"""

import librosa
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

def extract_time_domain_features(y, sr):
    """
    Extract time-domain features from audio signal.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        dict: Time-domain features
    """
    features = {}
    
    # Basic statistics
    features['rms'] = float(np.sqrt(np.mean(y**2)))
    features['energy'] = float(np.sum(y**2))
    features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    
    # Statistical moments
    features['mean'] = float(np.mean(y))
    features['std'] = float(np.std(y))
    features['skewness'] = float(stats.skew(y))
    features['kurtosis'] = float(stats.kurtosis(y))
    
    # Peak and amplitude features
    features['max_amplitude'] = float(np.max(np.abs(y)))
    features['min_amplitude'] = float(np.min(y))
    features['peak_to_peak'] = float(np.ptp(y))
    
    # Envelope features
    envelope = np.abs(librosa.util.frame(y, frame_length=2048, hop_length=512).max(axis=0))
    features['envelope_mean'] = float(np.mean(envelope))
    features['envelope_std'] = float(np.std(envelope))
    
    return features

def extract_frequency_domain_features(y, sr):
    """
    Extract frequency-domain features from audio signal.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        dict: Frequency-domain features
    """
    features = {}
    
    # Compute STFT
    stft = librosa.stft(y)
    magnitude = np.abs(stft)
    
    # Spectral features using librosa
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
    features['spectral_centroid_std'] = float(np.std(spectral_centroids))
    features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
    features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
    features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
    features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
    features['spectral_contrast_std'] = float(np.std(spectral_contrast))
    features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
    features['spectral_flatness_std'] = float(np.std(spectral_flatness))
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
        features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = float(np.mean(chroma))
    features['chroma_std'] = float(np.std(chroma))
    
    # Tonnetz (harmonic network)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    features['tonnetz_mean'] = float(np.mean(tonnetz))
    features['tonnetz_std'] = float(np.std(tonnetz))
    
    # Fundamental frequency
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if pitch_values:
        features['fundamental_freq_mean'] = float(np.mean(pitch_values))
        features['fundamental_freq_std'] = float(np.std(pitch_values))
    else:
        features['fundamental_freq_mean'] = 0.0
        features['fundamental_freq_std'] = 0.0
    
    return features

def extract_fault_specific_features(y, sr):
    """
    Extract features specifically relevant to motor fault detection.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        dict: Fault-specific features
    """
    features = {}
    
    # Harmonic-to-noise ratio
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(harmonic**2)
    noise_energy = np.sum(percussive**2)
    
    if noise_energy > 0:
        features['harmonic_noise_ratio'] = float(harmonic_energy / noise_energy)
    else:
        features['harmonic_noise_ratio'] = float('inf')
    
    # Spectral irregularity
    stft = librosa.stft(y)
    magnitude = np.abs(stft)
    spectral_irregularity = []
    
    for frame in magnitude.T:
        if len(frame) > 1:
            diff = np.diff(frame)
            irregularity = np.sum(np.abs(diff))
            spectral_irregularity.append(irregularity)
    
    features['spectral_irregularity_mean'] = float(np.mean(spectral_irregularity)) if spectral_irregularity else 0.0
    features['spectral_irregularity_std'] = float(np.std(spectral_irregularity)) if spectral_irregularity else 0.0
    
    # Low frequency energy ratio (for detecting bearing faults)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    magnitude_spectrum = np.mean(np.abs(librosa.stft(y, n_fft=2048)), axis=1)
    
    low_freq_mask = freqs <= 1000  # Below 1kHz
    high_freq_mask = freqs > 1000  # Above 1kHz
    
    low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask])
    total_energy = np.sum(magnitude_spectrum)
    
    features['low_freq_energy_ratio'] = float(low_freq_energy / total_energy) if total_energy > 0 else 0.0
    
    # Spectral peaks (for detecting gear faults)
    peaks = []
    for frame in np.abs(librosa.stft(y)).T:
        frame_peaks = []
        for i in range(1, len(frame)-1):
            if frame[i] > frame[i-1] and frame[i] > frame[i+1]:
                frame_peaks.append(frame[i])
        if frame_peaks:
            peaks.extend(frame_peaks)
    
    features['spectral_peaks_mean'] = float(np.mean(peaks)) if peaks else 0.0
    features['spectral_peaks_count'] = len(peaks)
    
    return features

def extract_all_features(audio_path):
    """
    Extract all features from audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        pandas.DataFrame: DataFrame with all extracted features
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract different types of features
    time_features = extract_time_domain_features(y, sr)
    freq_features = extract_frequency_domain_features(y, sr)
    fault_features = extract_fault_specific_features(y, sr)
    
    # Combine all features
    all_features = {**time_features, **freq_features, **fault_features}
    
    # Add metadata
    all_features['duration'] = float(len(y) / sr)
    all_features['sample_rate'] = int(sr)
    all_features['n_samples'] = len(y)
    
    # Convert to DataFrame
    df = pd.DataFrame([all_features])
    
    return df
