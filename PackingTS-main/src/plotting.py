import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_professional_result(file_id, raw_speed, true_segs, pred_segs, ignored_segs, fake_segs_pred, fake_segs_true, metrics, save_dir):
    fig, (ax_speed, ax_timeline) = plt.subplots(2, 1, figsize=(20, 8), 
                                                sharex=True, 
                                                gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.05})
    
    ax_speed.plot(raw_speed, color='#2c3e50', linewidth=1, alpha=0.8)
    ax_speed.fill_between(range(len(raw_speed)), raw_speed, color='#2c3e50', alpha=0.05)
    
    for s, e in pred_segs:
        ax_speed.axvline(s, color='#007bff', alpha=0.1, linestyle='--')
        ax_speed.axvline(e, color='#007bff', alpha=0.1, linestyle='--')

    ax_speed.set_ylabel('Speed', fontsize=10, weight='bold')
    title = (f"File: {file_id} | True: {metrics['n_true']} | Pred: {metrics['n_pred']} | "
             f"Diff: {metrics['diff']} | F1: {metrics['f1']:.2f}")
    ax_speed.set_title(title, fontsize=14, weight='bold', pad=15)
    
    y_true_pack, y_pred_pack, y_fakes, y_noise = 3, 2, 1, 0
    bar_height = 0.6
    
    for s, e in true_segs:
        ax_timeline.broken_barh([(s, e-s)], (y_true_pack - bar_height/2, bar_height), facecolors='#28a745', edgecolors='#1e7e34')
    for s, e in fake_segs_true:
        ax_timeline.broken_barh([(s, e-s)], (y_true_pack - bar_height/2, bar_height), facecolors='#fd7e14', edgecolors='#d9480f')
        
    for s, e in pred_segs:
        ax_timeline.broken_barh([(s, e-s)], (y_pred_pack - bar_height/2, bar_height), facecolors='#007bff', edgecolors='#0056b3')
        
    for s, e in fake_segs_pred:
        ax_timeline.broken_barh([(s, e-s)], (y_fakes - bar_height/2, bar_height), facecolors='#dc3545', edgecolors='#a71d2a')

    for s, e in ignored_segs:
        ax_timeline.broken_barh([(s, e-s)], (y_noise - bar_height/2, bar_height), facecolors='#ecf0f1', edgecolors='#95a5a6', hatch='///')

    ax_timeline.set_yticks([y_noise, y_fakes, y_pred_pack, y_true_pack])
    ax_timeline.set_yticklabels(['Ignored', 'Pred Fakes', 'Pred Packing', 'Ground Truth'], fontsize=9, weight='bold')
    ax_timeline.set_ylim(-0.6, 3.6)
    
    save_path = Path(save_dir) / f'final_viz_{file_id}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance(feature_df, save_dir):
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_df.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    save_path = Path(save_dir) / 'feature_importance.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_distributions(segments_data, save_dir):
    if not segments_data: return
    df = pd.DataFrame(segments_data)
    labels = df['label'].unique()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for label in labels:
        subset = df[df['label'] == label]
        axes[0].hist(subset['duration'], alpha=0.6, label=label, bins=15, density=True)
    axes[0].set_title('Duration Density')
    axes[0].legend()
    for label in labels:
        subset = df[df['label'] == label]
        axes[1].hist(subset['max_speed'], alpha=0.6, label=label, bins=15, density=True)
    axes[1].set_title('Max Speed Density')
    for label in labels:
        subset = df[df['label'] == label]
        axes[2].hist(subset['zero_ratio'], alpha=0.6, label=label, bins=15, density=True)
    axes[2].set_title('Zero Ratio Density')
    plt.tight_layout()
    save_path = Path(save_dir) / 'distributions.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix(y_true_labels, y_pred_labels, class_names, save_dir):
    """
    Визуализация матрицы ошибок
    
    Args:
        y_true_labels: истинные метки классов (список)
        y_pred_labels: предсказанные метки классов (список)
        class_names: названия классов
        save_dir: директория для сохранения
    """
    from sklearn.metrics import confusion_matrix as cm
    
    cm_matrix = cm(y_true_labels, y_pred_labels, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, weight='bold')
    plt.ylabel('True Label', fontsize=12, weight='bold')
    plt.xlabel('Predicted Label', fontsize=12, weight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()