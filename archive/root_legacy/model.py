"""
Архитектура нейронной сети для детекции артефактов в ЭКГ сигнале (PyTorch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ECGArtifactDetector(nn.Module):
    """
    Нейронная сеть для детекции артефактов в ЭКГ сигнале
    
    Классифицирует артефакты на:
    - 0: Нормальный сигнал
    - 1: Механический артефакт (шум, движение)
    - 2: Проблема с сердцем (аритмия, аномалии)
    """
    
    def __init__(self, input_length: int = 2000, num_classes: int = 3):
        """
        Инициализация модели
        
        Args:
            input_length: Длина входного сигнала (количество отсчётов)
            num_classes: Количество классов (норма, механический артефакт, проблема с сердцем)
        """
        super(ECGArtifactDetector, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # 1D Convolutional layers для извлечения признаков
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Attention mechanism для фокусировки на важных участках
        # Multi-head attention
        self.attention_dim = 64
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=4,
            batch_first=True
        )
        self.attention_proj = nn.Linear(128, self.attention_dim)
        
        # LSTM для учета временной зависимости
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=False)
        
        # Dropout для регуляризации
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        
        # Полносвязные слои
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Args:
            x: Входной тензор формы (batch_size, input_length) или (batch_size, input_length, 1)
            
        Returns:
            Выходной тензор формы (batch_size, num_classes)
        """
        # Добавляем размерность канала, если её нет
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, length)
        elif len(x.shape) == 3 and x.shape[1] != 1:
            # Если форма (batch, length, channels), транспонируем
            x = x.transpose(1, 2)
        
        # Convolutional блок 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Convolutional блок 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        
        # Convolutional блок 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Транспонируем для attention и LSTM: (batch, channels, length) -> (batch, length, channels)
        x = x.transpose(1, 2)
        
        # Attention mechanism
        x_proj = self.attention_proj(x)  # (batch, length, attention_dim)
        x_att, _ = self.attention(x_proj, x_proj, x_proj)
        x = x + x_att  # Residual connection
        
        # LSTM блоки
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Берем последний выход LSTM
        x = x[:, -1, :]  # (batch, hidden_size)
        
        # Полносвязные слои
        x = self.dropout2(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        output = self.output_layer(x)
        
        return output


class LightweightECGDetector(nn.Module):
    """
    Облегченная модель для мобильных устройств
    """
    
    def __init__(self, input_length: int = 2000, num_classes: int = 3):
        """
        Инициализация облегченной модели
        
        Args:
            input_length: Длина входного сигнала
            num_classes: Количество классов
        """
        super(LightweightECGDetector, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # Упрощенная архитектура для мобильных устройств
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Полносвязные слои
        self.dense1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, num_classes)
    
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Args:
            x: Входной тензор формы (batch_size, input_length) или (batch_size, input_length, 1)
            
        Returns:
            Выходной тензор формы (batch_size, num_classes)
        """
        # Добавляем размерность канала, если её нет
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, length)
        elif len(x.shape) == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        
        # Convolutional блоки
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        
        # Полносвязные слои
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        output = self.output_layer(x)
        
        return output


def create_lightweight_model(input_length: int = 2000, num_classes: int = 3) -> LightweightECGDetector:
    """
    Создание облегченной модели для мобильных устройств
    
    Args:
        input_length: Длина входного сигнала
        num_classes: Количество классов
        
    Returns:
        Облегченная модель
    """
    return LightweightECGDetector(input_length, num_classes)
