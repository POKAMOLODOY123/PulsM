from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_ecg_with_artifacts(
    ecg: np.ndarray,
    *,
    fs: float = 125.0,
    artifact_mask: Optional[np.ndarray] = None,
    normal_mask: Optional[np.ndarray] = None,
    title: str = "ECG with artifact marking",
    figsize: Sequence[float] = (14, 4),
) -> plt.Figure:
    """
    Рисует график ЭКГ “как в Novaroll”:
    - сигнал,
    - зелёная подложка где normal_mask == True,
    - красная подложка где artifact_mask == True.

    Args:
        ecg: 1‑канальный сигнал формы (n_samples,)
        fs: частота дискретизации (Гц)
        artifact_mask: булева маска артефактов (True = артефакт)
        normal_mask: булева маска нормальных участков (True = норма)
        title: заголовок графика
        figsize: размер фигуры matplotlib

    Returns:
        figure matplotlib, чтобы можно было сохранить или показать.
    """
    x = np.asarray(ecg, dtype=float).reshape(-1)
    n = x.shape[0]
    t = np.arange(n) / float(fs)

    if artifact_mask is not None:
        artifact_mask = np.asarray(artifact_mask, dtype=bool).reshape(-1)
        if artifact_mask.shape[0] != n:
            raise ValueError("artifact_mask должен иметь ту же длину, что и ecg")
    if normal_mask is not None:
        normal_mask = np.asarray(normal_mask, dtype=bool).reshape(-1)
        if normal_mask.shape[0] != n:
            raise ValueError("normal_mask должен иметь ту же длину, что и ecg")

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)

    # Сама кривая ЭКГ
    ax.plot(t, x, color="black", linewidth=0.8, label="ECG")

    ymin = np.min(x)
    ymax = np.max(x)
    yrange = ymax - ymin if ymax > ymin else 1.0
    pad = 0.05 * yrange
    ymin_plot = ymin - pad
    ymax_plot = ymax + pad

    def _fill_regions(mask: np.ndarray, color: str, alpha: float) -> None:
        """Подсветка регионов по булевой маске."""
        if mask is None or not mask.any():
            return
        mask = mask.astype(bool)
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if mask[0]:
            starts = np.concatenate(([0], starts))
        if mask[-1]:
            ends = np.concatenate((ends, [mask.size - 1]))
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[e], color=color, alpha=alpha, linewidth=0)

    # Зелёная область — норма
    if normal_mask is not None:
        _fill_regions(normal_mask, color="#00cc66", alpha=0.15)

    # Красная область — артефакты
    if artifact_mask is not None:
        _fill_regions(artifact_mask, color="#ff3333", alpha=0.25)

    ax.set_ylim(ymin_plot, ymax_plot)
    ax.set_xlabel("Time, s")
    ax.set_ylabel("ECG amplitude")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_ecg_debug(
    ecg: np.ndarray,
    *,
    fs: float = 125.0,
    artifact_mask: Optional[np.ndarray] = None,
    normal_mask: Optional[np.ndarray] = None,
    artifact_probs: Optional[np.ndarray] = None,
    artifact_probs_smoothed: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
    feature_var: Optional[np.ndarray] = None,
    feature_mean_abs_grad: Optional[np.ndarray] = None,
    title: str = "ECG debug view",
    figsize: Sequence[float] = (14, 8),
) -> plt.Figure:
    """
    Debug-визуализация:
    - ЭКГ + зелёная/красная подложка (как в Novaroll),
    - скоринг/вероятности артефакта (сырой и сглаженный),
    - несколько “объясняющих” фич (std/var/mean|grad|) для выбранного окна.
    """
    x = np.asarray(ecg, dtype=float).reshape(-1)
    n = x.shape[0]
    t = np.arange(n) / float(fs)

    def _as_vec(v: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
        if v is None:
            return None
        vv = np.asarray(v).reshape(-1)
        if vv.shape[0] != n:
            raise ValueError(f"{name} должен иметь длину как ecg")
        return vv

    artifact_mask = _as_vec(artifact_mask, "artifact_mask").astype(bool) if artifact_mask is not None else None
    normal_mask = _as_vec(normal_mask, "normal_mask").astype(bool) if normal_mask is not None else None
    artifact_probs = _as_vec(artifact_probs, "artifact_probs").astype(float) if artifact_probs is not None else None
    artifact_probs_smoothed = (
        _as_vec(artifact_probs_smoothed, "artifact_probs_smoothed").astype(float)
        if artifact_probs_smoothed is not None
        else None
    )
    feature_std = _as_vec(feature_std, "feature_std").astype(float) if feature_std is not None else None
    feature_var = _as_vec(feature_var, "feature_var").astype(float) if feature_var is not None else None
    feature_mean_abs_grad = (
        _as_vec(feature_mean_abs_grad, "feature_mean_abs_grad").astype(float)
        if feature_mean_abs_grad is not None
        else None
    )

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.08})
    ax_sig, ax_prob, ax_feat = axes

    ax_sig.plot(t, x, color="black", linewidth=0.8, label="ECG")

    def _fill_regions(ax, mask: np.ndarray, color: str, alpha: float) -> None:
        if mask is None or not mask.any():
            return
        m = mask.astype(bool)
        diff = np.diff(m.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if m[0]:
            starts = np.concatenate(([0], starts))
        if m[-1]:
            ends = np.concatenate((ends, [m.size - 1]))
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[e], color=color, alpha=alpha, linewidth=0)

    if normal_mask is not None:
        _fill_regions(ax_sig, normal_mask, color="#00cc66", alpha=0.12)
    if artifact_mask is not None:
        _fill_regions(ax_sig, artifact_mask, color="#ff3333", alpha=0.20)

    ax_sig.set_ylabel("ECG")
    ax_sig.grid(True, linestyle="--", alpha=0.25)
    ax_sig.legend(loc="upper right")

    if artifact_probs is not None:
        ax_prob.plot(t, artifact_probs, color="#6c757d", linewidth=1.0, label="artifact_probs")
    if artifact_probs_smoothed is not None:
        ax_prob.plot(t, artifact_probs_smoothed, color="#007bff", linewidth=1.2, label="artifact_probs_smoothed")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("Prob")
    ax_prob.grid(True, linestyle="--", alpha=0.25)
    ax_prob.legend(loc="upper right")

    # Feature panel (scaled for readability)
    if feature_std is not None:
        ax_feat.plot(t, feature_std, color="#17a2b8", linewidth=1.0, label="roll_std")
    if feature_var is not None:
        ax_feat.plot(t, feature_var, color="#6610f2", linewidth=1.0, alpha=0.85, label="roll_var")
    if feature_mean_abs_grad is not None:
        ax_feat.plot(t, feature_mean_abs_grad, color="#fd7e14", linewidth=1.0, label="roll_mean_abs_grad")
    ax_feat.set_ylabel("Features")
    ax_feat.set_xlabel("Time, s")
    ax_feat.grid(True, linestyle="--", alpha=0.25)
    ax_feat.legend(loc="upper right")

    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    return fig


def apply_mask_for_metrics(
    ecg: np.ndarray,
    *,
    artifact_mask: np.ndarray,
) -> np.ndarray:
    """
    Пример функции “выкинуть красное перед метриками”.

    На практике ты можешь:
    - либо просто обнулять артефактные точки,
    - либо разбивать сигнал на куски по normal_mask и считать метрики по каждому.

    Здесь реализован простой вариант: артефактные точки зануляем.
    """
    x = np.asarray(ecg, dtype=float).reshape(-1)
    mask = np.asarray(artifact_mask, dtype=bool).reshape(-1)
    if x.shape[0] != mask.shape[0]:
        raise ValueError("artifact_mask должен иметь ту же длину, что и ecg")

    clean = x.copy()
    clean[mask] = 0.0
    return clean


__all__ = ["plot_ecg_with_artifacts", "plot_ecg_debug", "apply_mask_for_metrics"]

