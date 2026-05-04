"""
Генерация синтетических механических артефактов для ЭКГ
"""
import numpy as np
from typing import Tuple, List
from scipy.signal import butter, filtfilt
import random


def add_baseline_wander(signal: np.ndarray, fs: int = 200) -> np.ndarray:
    """Добавляет дрейф изолинии (движение пациента)"""
    t = np.arange(len(signal)) / fs
    # Низкочастотный синус (0.1-0.5 Гц)
    freq = random.uniform(0.1, 0.5)
    amplitude = random.uniform(0.1, 0.3) * np.std(signal)
    baseline_drift = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + baseline_drift


def add_muscle_noise(signal: np.ndarray, fs: int = 200) -> np.ndarray:
    """Добавляет мышечные артефакты (20-100 Гц)"""
    # Высокочастотный шум
    noise_power = random.uniform(0.05, 0.2) * np.std(signal)
    muscle_noise = noise_power * np.random.randn(len(signal))
    
    # Фильтруем в диапазоне мышечных артефактов
    nyquist = fs / 2
    low = 20 / nyquist
    high = min(100 / nyquist, 0.99)
    
    if low < high:
        b, a = butter(4, [low, high], btype='band')
        muscle_noise = filtfilt(b, a, muscle_noise)
    
    return signal + muscle_noise


def add_electrode_noise(signal: np.ndarray) -> np.ndarray:
    """Добавляет шум от плохого контакта электродов"""
    # Случайные импульсы
    noise = np.zeros_like(signal)
    n_spikes = random.randint(1, 5)
    
    for _ in range(n_spikes):
        pos = random.randint(0, len(signal) - 1)
        amplitude = random.uniform(0.5, 2.0) * np.std(signal)
        width = random.randint(5, 20)
        
        start = max(0, pos - width // 2)
        end = min(len(signal), pos + width // 2)
        noise[start:end] += amplitude * np.exp(-((np.arange(end - start) - width // 2) ** 2) / (width / 4) ** 2)
    
    return signal + noise


def add_powerline_interference(signal: np.ndarray, fs: int = 200) -> np.ndarray:
    """Добавляет сетевую наводку 50/60 Гц"""
    t = np.arange(len(signal)) / fs
    freq = random.choice([50, 60])  # Европа/США
    amplitude = random.uniform(0.05, 0.15) * np.std(signal)
    interference = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + interference


def generate_mechanical_artifacts(
    clean_segments: np.ndarray, 
    n_artifacts: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует механические артефакты из чистых сегментов
    
    Args:
        clean_segments: Чистые ЭКГ сегменты формы (n_segments, length)
        n_artifacts: Количество артефактов для генерации
        
    Returns:
        Tuple (артефакты, метки)
    """
    if n_artifacts is None:
        n_artifacts = len(clean_segments) // 3  # 1/3 от количества чистых
    
    artifacts = []
    labels = []
    
    artifact_functions = [
        add_baseline_wander,
        add_muscle_noise, 
        add_electrode_noise,
        add_powerline_interference
    ]
    
    for i in range(n_artifacts):
        # Выбираем случайный чистый сегмент
        base_idx = random.randint(0, len(clean_segments) - 1)
        base_signal = clean_segments[base_idx].copy()
        
        # Применяем 1-3 типа артефактов
        n_noise_types = random.randint(1, 3)
        selected_functions = random.sample(artifact_functions, n_noise_types)
        
        noisy_signal = base_signal
        for func in selected_functions:
            noisy_signal = func(noisy_signal)
        
        artifacts.append(noisy_signal)
        labels.append(1)  # Mechanical Artifact
    
    return np.array(artifacts), np.array(labels)


def create_balanced_dataset(
    normal_segments: np.ndarray,
    heart_issue_segments: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создаёт сбалансированный датасет с синтетическими артефактами
    
    Returns:
        Tuple (все_сегменты, все_метки)
    """
    # Генерируем механические артефакты
    n_artifacts = min(len(normal_segments), len(heart_issue_segments))
    source_segments = np.concatenate([normal_segments, heart_issue_segments])
    
    artifact_segments, artifact_labels = generate_mechanical_artifacts(
        source_segments, n_artifacts
    )
    
    # Объединяем все данные
    all_segments = np.concatenate([
        normal_segments,
        artifact_segments, 
        heart_issue_segments
    ])
    
    all_labels = np.concatenate([
        np.zeros(len(normal_segments)),      # 0: Normal
        artifact_labels,                     # 1: Mechanical Artifact  
        np.full(len(heart_issue_segments), 2) # 2: Heart Issue
    ])
    
    return all_segments, all_labels







