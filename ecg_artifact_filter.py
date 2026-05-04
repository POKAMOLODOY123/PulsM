import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - опциональная зависимость
    joblib = None


ArrayLike = np.ndarray
Segment = Tuple[int, int]


@dataclass
class ECGArtifactFilterConfig:
    """
    Конфиг постобработки и фичей для фильтрации ЭКГ.

    Все параметры заданы в отсчётах, т.к. частота дискретизации фиксирована (125 Гц).
    При желании можно пересчитать из секунд: samples = seconds * 125.
    """

    # Размеры окон (в отсчётах) для расчёта статистик.
    # Для 125 Гц:
    # - 125 ~ 1.0 c
    # - 188 ~ 1.5 c (1500 мс)
    window_sizes: Tuple[int, ...] = (5, 15, 31, 63, 125, 188)

    # Порог вероятности / скоринга артефакта
    prob_threshold: float = 0.5

    # Сглаживание вероятностей перед порогом (rolling mean, в отсчётах).
    # Помогает не дробить сегменты на шумных прогнозах.
    prob_smooth_window: int = 1

    # Минимальная длительность артефакта (в отсчётах), чтобы считать его настоящим
    min_artifact_duration: int = 25  # ~0.2 c при 125 Гц

    # Максимальная длина "дыры" между двумя кусками артефакта (заполняется как артефакт)
    gap_tolerance: int = 10  # ~0.08 с

    # Частота дискретизации сигнала (на будущее, если понадобится пересчёт)
    fs: int = 125

    # Нормирование входного сигнала перед фичами/детекцией.
    # Помогает переносимости между пользователями и устройствами.
    normalize_input: bool = True

    # Добавлять ли дисперсию (variance) как отдельную фичу.
    # Нужен флаг для совместимости со старыми моделями (ожидают меньше фич).
    include_variance: bool = True

    # Правило “провал/выброс относительно локальной базовой линии”.
    # Помогает, когда модель/эвристика не видит явные провалы амплитуды.
    use_baseline_deviation_rule: bool = False
    baseline_window: int = 125  # ~1s
    baseline_std_k: float = 4.0

    # Правило “залипание / плато / сильная квантизация”:
    # если долго подряд почти нет изменений (много одинаковых/почти одинаковых значений),
    # то это технический артефакт (потеря контакта/АЦП/передача).
    use_plateau_rule: bool = False
    plateau_window: int = 125  # ~1s
    plateau_eps: float = 0.0  # 0.0 для точного равенства; можно поставить 1e-3 для float шума
    plateau_min_ratio: float = 0.98  # доля “почти нулевых” d/dt в окне

    # Правило “flatline”: локальная дисперсия слишком мала.
    use_flatline_rule: bool = False
    flatline_window: int = 125  # ~1s
    flatline_std_threshold: float = 0.02  # в нормированных единицах (после normalize_input)

    # Multi-scale постпроцессинг (как в PackingTS, но для ЭКГ).
    # Если включен, сегменты ищутся для нескольких значений min_artifact_duration
    # и затем объединяются.
    use_multi_scale: bool = False
    multi_scale_min_durations: Tuple[int, ...] = (50, 125, 250)


def _rolling_mean(x: ArrayLike, w: int) -> ArrayLike:
    if w <= 1:
        return x.astype(float)
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")


def _rolling_std(x: ArrayLike, w: int) -> ArrayLike:
    if w <= 1:
        return np.zeros_like(x, dtype=float)
    mean = _rolling_mean(x, w)
    mean_sq = _rolling_mean(x * x, w)
    var = np.clip(mean_sq - mean * mean, a_min=0.0, a_max=None)
    return np.sqrt(var)


def _build_ecg_features(
    signal: ArrayLike, window_sizes: Tuple[int, ...], *, include_variance: bool = True
) -> ArrayLike:
    """
    Строит простые табличные фичи по 1‑канальному ЭКГ (аналогично тому,
    как в PackingTS считаются rolling‑статистики, только без pandas).

    Фичи на каждую точку:
    - raw, |raw|
    - d/dt raw, |d/dt raw|
    - rolling mean/std/var по нескольким окнам
    """
    if signal.ndim != 1:
        raise ValueError("Ожидается одномерный сигнал формы (n_samples,)")

    x = signal.astype(float)
    n = x.shape[0]

    # Базовые ряды
    abs_x = np.abs(x)
    # Простейшая производная
    grad = np.empty_like(x)
    if n > 1:
        grad[0] = x[1] - x[0]
        grad[1:] = x[1:] - x[:-1]
    else:
        grad.fill(0.0)
    abs_grad = np.abs(grad)

    features: List[ArrayLike] = [x, abs_x, grad, abs_grad]

    for w in window_sizes:
        mean_w = _rolling_mean(x, w)
        std_w = _rolling_std(x, w)
        mean_abs_grad_w = _rolling_mean(abs_grad, w)
        if include_variance:
            var_w = std_w * std_w
            features.extend([mean_w, std_w, var_w, mean_abs_grad_w])
        else:
            features.extend([mean_w, std_w, mean_abs_grad_w])

    # Стекаем в матрицу (n_samples, n_features)
    feats = np.stack(features, axis=1)
    # На всякий случай защищаемся от NaN/inf
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def _find_segments_from_mask(mask: ArrayLike) -> List[Segment]:
    """
    Находит [start, end] по булевой маске без scipy.ndimage.
    """
    if mask.size == 0:
        return []

    m = mask.astype(bool)
    diff = np.diff(m.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if m[0]:
        starts = np.concatenate(([0], starts))
    if m[-1]:
        ends = np.concatenate((ends, [m.size - 1]))

    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _fill_small_gaps(mask: ArrayLike, max_gap: int) -> ArrayLike:
    """
    Заполняет короткие "нули" между единицами (дыры меньше max_gap считаем частью артефакта).
    """
    if max_gap <= 0 or mask.size == 0:
        return mask.astype(bool)

    m = mask.astype(bool)
    inv = ~m
    zero_segments = _find_segments_from_mask(inv)

    m_filled = m.copy()
    for s, e in zero_segments:
        if (e - s + 1) <= max_gap:
            m_filled[s : e + 1] = True
    return m_filled


def _segments_from_probs(
    probs: ArrayLike, config: ECGArtifactFilterConfig
) -> Tuple[ArrayLike, List[Segment]]:
    """
    Простая постобработка, вдохновлённая smart_event_counter_v3:
    - порог по вероятности/скорингу
    - склейка через короткие дыры
    - отсечение слишком коротких сегментов
    """
    if probs.ndim != 1:
        raise ValueError("probs должен быть одномерным вектором формы (n_samples,)")

    # Сглаживаем вероятности, чтобы убрать “дребезг” около порога.
    w = int(getattr(config, "prob_smooth_window", 1))
    probs_sm = _rolling_mean(probs.astype(float), w) if w and w > 1 else probs.astype(float)

    def _one_scale(min_duration: int) -> Tuple[ArrayLike, List[Segment]]:
        base_mask = probs_sm >= float(config.prob_threshold)
        mask_with_gaps_filled = _fill_small_gaps(base_mask, config.gap_tolerance)
        raw_segments = _find_segments_from_mask(mask_with_gaps_filled)
        m = np.zeros_like(mask_with_gaps_filled, dtype=bool)
        segs: List[Segment] = []
        for s, e in raw_segments:
            if (e - s + 1) >= min_duration:
                m[s : e + 1] = True
                segs.append((s, e))
        return m, segs

    if getattr(config, "use_multi_scale", False):
        all_segments: List[Segment] = []
        min_durs = list(getattr(config, "multi_scale_min_durations", (config.min_artifact_duration,)))
        for d in min_durs:
            _, segs = _one_scale(int(d))
            all_segments.extend(segs)

        if not all_segments:
            return np.zeros_like(probs_sm, dtype=bool), []

        # Объединяем перекрывающиеся сегменты (multi-scale union)
        all_segments = sorted(all_segments, key=lambda x: x[0])
        merged: List[Segment] = []
        cur_s, cur_e = all_segments[0]
        for s, e in all_segments[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        final_mask = np.zeros_like(probs_sm, dtype=bool)
        for s, e in merged:
            final_mask[s : e + 1] = True
        return final_mask, merged

    # Обычный одно-масштабный режим
    final_mask, kept_segments = _one_scale(config.min_artifact_duration)
    return final_mask, kept_segments


def _baseline_deviation_mask(x: ArrayLike, window: int, k: float) -> ArrayLike:
    """
    True там, где сигнал сильно отклоняется от локальной базовой линии:
    |x - mean_w| > k * std_w.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    w = int(window)
    if x.size == 0 or w <= 1:
        return np.zeros_like(x, dtype=bool)
    mean_w = _rolling_mean(x, w)
    std_w = _rolling_std(x, w)
    std_w = np.where(std_w < 1e-6, 1e-6, std_w)
    z = np.abs(x - mean_w) / std_w
    return z > float(k)


def _plateau_mask(x: ArrayLike, window: int, eps: float, min_ratio: float) -> ArrayLike:
    """
    True там, где в окне доля “почти нулевых” приращений высокая.
    Для браслета это часто ловит потерю контакта/залипание/квантизацию.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    w = int(window)
    if n == 0 or w <= 2:
        return np.zeros_like(x, dtype=bool)

    dx = np.empty_like(x)
    if n > 1:
        dx[0] = 0.0
        dx[1:] = x[1:] - x[:-1]
    else:
        dx.fill(0.0)

    still = (np.abs(dx) <= float(eps)).astype(float)
    ratio = _rolling_mean(still, w)
    return ratio >= float(min_ratio)


def _flatline_mask(x: ArrayLike, window: int, std_threshold: float) -> ArrayLike:
    """
    True там, где локальный std слишком мал (сигнал “прибит”).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    w = int(window)
    if x.size == 0 or w <= 2:
        return np.zeros_like(x, dtype=bool)
    std_w = _rolling_std(x, w)
    return std_w <= float(std_threshold)

def _heuristic_artifact_score(ecg: ArrayLike, window: int = 25) -> ArrayLike:
    """
    Эвристический скоринг артефактов на основе энергии производной.

    Нужен, чтобы модуль уже сейчас что‑то делал без обученной модели.
    """
    x = ecg.astype(float)
    if x.ndim != 1:
        raise ValueError("Ожидается одномерный ЭКГ сигнал")

    # Уберём DC-смещение
    x = x - np.median(x)

    # Производная и её модуль
    grad = np.empty_like(x)
    if x.size > 1:
        grad[0] = x[1] - x[0]
        grad[1:] = x[1:] - x[:-1]
    else:
        grad.fill(0.0)
    energy = np.abs(grad)

    # Сглаживание энергии
    energy_smooth = _rolling_mean(energy, max(3, int(window)))

    # Нормализация в [0, 1] по робастному масштабу
    p95 = np.percentile(energy_smooth, 95) if energy_smooth.size > 0 else 1.0
    scale = p95 if p95 > 1e-6 else 1.0
    score = np.clip(energy_smooth / scale, 0.0, 3.0) / 3.0
    return score.astype(float)


class ECGArtifactFilter:
    """
    Модуль фильтрации артефактов ЭКГ.

    Вход: 1‑канальный ЭКГ (np.ndarray, 125 Гц).
    Выход: разметка нормальных/аномальных точек + список аномальных сегментов.

    Можно использовать:
    - либо с обученной моделью (RandomForest / любая sklearn‑совместимая модель),
    - либо без модели — тогда используется эвристика по энергии производной.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[ECGArtifactFilterConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or ECGArtifactFilterConfig()

    @classmethod
    def from_joblib(
        cls,
        model_path: str,
        config: Optional[ECGArtifactFilterConfig] = None,
    ) -> "ECGArtifactFilter":
        """
        Загружает sklearn‑модель из .joblib файла.
        """
        if joblib is None:
            raise ImportError(
                "Для загрузки модели нужен пакет 'joblib'. "
                "Установите его или создайте ECGArtifactFilter(model=...) вручную."
            )
        model = joblib.load(model_path)
        # На всякий случай ограничим число потоков
        if hasattr(model, "set_params"):
            try:
                model.set_params(n_jobs=1)
            except Exception:
                pass
        return cls(model=model, config=config)

    def _predict_artifact_proba(self, ecg: ArrayLike) -> ArrayLike:
        """
        Возвращает вектор "вероятности" артефакта на каждую точку.

        - если есть модель с predict_proba: берём вероятность "артефакт"‑класса.
        - если модели нет: используем эвристику _heuristic_artifact_score.
        """
        x = np.asarray(ecg, dtype=float)
        if x.ndim != 1:
            raise ValueError("Ожидается одномерный ЭКГ сигнал формы (n_samples,)")

        if getattr(self.config, "normalize_input", False):
            # Робастное центрирование + масштабирование.
            # Используем std, чтобы не вводить новые зависимости.
            x = x - np.median(x)
            s = float(np.std(x))
            if s > 1e-6:
                x = x / s

        if self.model is None:
            return _heuristic_artifact_score(x)

        # Табличные фичи, аналогично PackingTS (но адаптированы под ЭКГ)
        feats = _build_ecg_features(
            x,
            self.config.window_sizes,
            include_variance=getattr(self.config, "include_variance", True),
        )
        if not hasattr(self.model, "predict_proba"):
            # fallback: если модель не умеет predict_proba, считаем, что она отдаёт логиты/скоринг
            raw_scores = self.model.predict(feats)  # type: ignore[no-any-return]
            raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
            # нормализуем в [0, 1] через сигмоиду
            return 1.0 / (1.0 + np.exp(-raw_scores))

        proba = self.model.predict_proba(feats)  # type: ignore[no-any-return]
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2 or proba.shape[0] != x.shape[0]:
            raise RuntimeError("Модель вернула неожиданный формат вероятностей")

        # Два случая:
        # - бинарная классификация: столбец 1 — класс 'артефакт'
        # - мультикласс: считаем, что класс 0 — 'норма', остальные — разные типы артефактов
        if proba.shape[1] == 2:
            return proba[:, 1]
        elif proba.shape[1] > 2:
            return np.clip(proba[:, 1:].sum(axis=1), 0.0, 1.0)
        else:
            # На всякий случай, если модель обучена как регрессор с 1 выходом
            return np.clip(proba.reshape(-1), 0.0, 1.0)

    def infer(self, ecg: ArrayLike) -> Dict[str, Any]:
        """
        Основной метод инференса.

        Args:
            ecg: np.ndarray формы (n_samples,) — 1‑канальный ЭКГ 125 Гц.

        Returns:
            dict с полями:
            - 'artifact_probs': np.ndarray(float32) длины n_samples, скоринг артефакта.
            - 'artifact_mask': np.ndarray(bool) длины n_samples — True там, где артефакт.
            - 'normal_mask': np.ndarray(bool) длины n_samples — True там, где сигнал нормальный.
            - 'artifact_segments': List[Tuple[int, int]] — сегменты артефактов [start, end].
        """
        x = np.asarray(ecg, dtype=float).reshape(-1)
        probs = self._predict_artifact_proba(x)
        artifact_mask, segments = _segments_from_probs(probs, self.config)

        # Дополнительные “хард-правила” (объединяем как OR и затем прогоняем постпроцессинг).
        extra_masks: List[np.ndarray] = []

        if getattr(self.config, "use_baseline_deviation_rule", False):
            w = int(getattr(self.config, "baseline_window", self.config.fs))
            k = float(getattr(self.config, "baseline_std_k", 4.0))
            extra_masks.append(_baseline_deviation_mask(x, window=w, k=k))

        if getattr(self.config, "use_plateau_rule", False):
            w = int(getattr(self.config, "plateau_window", self.config.fs))
            eps = float(getattr(self.config, "plateau_eps", 0.0))
            r = float(getattr(self.config, "plateau_min_ratio", 0.98))
            extra_masks.append(_plateau_mask(x, window=w, eps=eps, min_ratio=r))

        if getattr(self.config, "use_flatline_rule", False):
            w = int(getattr(self.config, "flatline_window", self.config.fs))
            thr = float(getattr(self.config, "flatline_std_threshold", 0.02))
            extra_masks.append(_flatline_mask(x, window=w, std_threshold=thr))

        if extra_masks:
            rule_mask = np.zeros_like(artifact_mask, dtype=bool)
            for m in extra_masks:
                rule_mask |= np.asarray(m, dtype=bool)

            if rule_mask.any():
                merged = artifact_mask | rule_mask
                merged = _fill_small_gaps(merged, self.config.gap_tolerance)
                raw_segments = _find_segments_from_mask(merged)
                artifact_mask = np.zeros_like(merged, dtype=bool)
                segments = []
                for s, e in raw_segments:
                    if (e - s + 1) >= self.config.min_artifact_duration:
                        artifact_mask[s : e + 1] = True
                        segments.append((s, e))

        normal_mask = ~artifact_mask

        return {
            "artifact_probs": probs.astype(np.float32),
            "artifact_mask": artifact_mask,
            "normal_mask": normal_mask,
            "artifact_segments": segments,
        }


__all__ = ["ECGArtifactFilterConfig", "ECGArtifactFilter"]

