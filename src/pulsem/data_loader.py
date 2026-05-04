"""
ECG signal loading and preprocessing utilities.
"""
import json
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt


class ECGDataLoader:
    """Helper class that loads and preprocesses ECG time series."""
    
    def __init__(self, sampling_rate: int = 200):
        """
        Args:
            sampling_rate: Target sampling rate (Hz) used for preprocessing.
        """
        self.sampling_rate = sampling_rate
    
    def load_csv(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load ECG values from a CSV file exported by a recorder.
        
        Args:
            file_path: Path to the CSV file.
        
        Returns:
            Tuple of raw signal values and detected sampling rate.
        """
        df = pd.read_csv(file_path)
        
        # Detect sampling rate from the column if it exists.
        sampling_rate_series = df['samplingRateHz'].dropna()
        if not sampling_rate_series.empty:
            sampling_rate = int(sampling_rate_series.iloc[0])
        else:
            sampling_rate = self.sampling_rate  # fallback when the column is empty
        
        # Skip header rows and keep the signal values
        ecg_signal = df['values'].iloc[2:].values.astype(np.float32)
        
        # Drop NaN values
        ecg_signal = ecg_signal[~np.isnan(ecg_signal)]
        
        return ecg_signal, sampling_rate

    def load_json(
        self,
        file_path: str,
        value_key: str = "values",
        fs_key: str = "samplingRateHz",
        default_fs: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load ECG values from a JSON file of the form:
            { "values": [...], "samplingRateHz": 125 }
        
        Args:
            file_path: Path to the JSON file.
            value_key: Key with ECG samples array.
            fs_key: Key with sampling rate (Hz).
            default_fs: Fallback sampling rate if key is missing.
        
        Returns:
            Tuple of raw signal values and detected sampling rate.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object with ECG data")
        
        if value_key not in data:
            raise KeyError(f"Key '{value_key}' not found in JSON")
        
        values = np.asarray(data[value_key], dtype=np.float32)
        fs = data.get(fs_key, default_fs if default_fs is not None else self.sampling_rate)
        fs = int(fs)
        
        # Drop NaN values just in case
        values = values[~np.isnan(values)]
        
        return values, fs
    
    def apply_bandpass_filter(self, signal_data: np.ndarray, 
                              lowcut: float = 0.5, 
                              highcut: float = 40.0) -> np.ndarray:
        """
        Apply a Butterworth band-pass filter to suppress baseline drift and high-frequency noise.

        Args:
            signal_data: Input ECG signal.
            lowcut: Lower cut-off frequency (Hz).
            highcut: Upper cut-off frequency (Hz).

        Returns:
            Filtered signal.
        """
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Normalize the signal using z-score normalization.

        Args:
            signal_data: Input ECG signal.

        Returns:
            Normalized signal.
        """
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        
        if std == 0:
            return signal_data - mean
        
        normalized = (signal_data - mean) / std
        return normalized
    
    def segment_signal(self, signal_data: np.ndarray, 
                       window_size: int = 2000, 
                       overlap: float = 0.5) -> List[np.ndarray]:
        """
        Split the time series into overlapping windows.

        Args:
            signal_data: Input ECG signal.
            window_size: Number of samples per window.
            overlap: Overlap ratio between consecutive windows (0.0 - 1.0).

        Returns:
            List of signal segments with equal length.
        """
        segments = []
        step = int(window_size * (1 - overlap))
        
        for i in range(0, len(signal_data) - window_size + 1, step):
            segment = signal_data[i:i + window_size]
            segments.append(segment)
        
        return segments
    
    def preprocess_signal(self, signal_data: np.ndarray, 
                          apply_filter: bool = True,
                          normalize: bool = True) -> np.ndarray:
        """
        Apply filtering and normalization in one call.

        Args:
            signal_data: Input ECG signal.
            apply_filter: Whether to run band-pass filtering.
            normalize: Whether to run z-score normalization.

        Returns:
            Clean signal ready for segmentation.
        """
        processed = signal_data.copy()
        
        if apply_filter:
            processed = self.apply_bandpass_filter(processed)
        
        if normalize:
            processed = self.normalize_signal(processed)
        
        return processed

