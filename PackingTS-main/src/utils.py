import logging, os, sys, numpy as np, pandas as pd
from tqdm import tqdm

def setup_logger(save_path, name="logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fh = logging.FileHandler(save_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def calculate_event_metrics(pred_segs, true_segs):
    tp = 0
    for ps, pe in pred_segs:
        for ts, te in true_segs:
            start, end = max(ps, ts), min(pe, te)
            if end > start:
                if (end-start)/(pe-ps) > 0.2 or (end-start)/(te-ts) > 0.2:
                    tp += 1
                    break
    prec = tp / len(pred_segs) if pred_segs else 0
    rec = tp / len(true_segs) if true_segs else 0
    return prec, rec, (2*prec*rec/(prec+rec) if prec+rec else 0)

def calculate_per_class_metrics(pred_segs_by_class, true_segs_by_class):
    """
    Рассчитывает метрики для каждого класса отдельно
    
    Args:
        pred_segs_by_class: dict {'Success': [...], 'Struggling': [...], 'Fake': [...]}
        true_segs_by_class: dict {'Success': [...], 'Struggling': [...], 'Fake': [...]}
    
    Returns:
        dict с метриками для каждого класса
    """
    metrics_by_class = {}
    
    for class_name in ['Success', 'Struggling', 'Fake']:
        pred_segs = pred_segs_by_class.get(class_name, [])
        true_segs = true_segs_by_class.get(class_name, [])
        
        prec, rec, f1 = calculate_event_metrics(pred_segs, true_segs)
        
        tp = 0
        for ps, pe in pred_segs:
            for ts, te in true_segs:
                start, end = max(ps, ts), min(pe, te)
                if end > start:
                    if (end-start)/(pe-ps) > 0.2 or (end-start)/(te-ts) > 0.2:
                        tp += 1
                        break
        
        fp = len(pred_segs) - tp
        fn = len(true_segs) - tp
        
        metrics_by_class[class_name] = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_pred': len(pred_segs),
            'n_true': len(true_segs)
        }
    
    return metrics_by_class

def calculate_temporal_iou(pred_segs, true_segs):
    """
    Рассчитывает Temporal IoU (Intersection over Union) для более строгой оценки локализации
    
    Returns:
        dict с IoU для каждого предсказанного сегмента
    """
    ious = []
    matched_true = set()
    
    for ps, pe in pred_segs:
        best_iou = 0.0
        best_match = None
        
        for idx, (ts, te) in enumerate(true_segs):
            if idx in matched_true:
                continue
            
            # Пересечение
            inter_start = max(ps, ts)
            inter_end = min(pe, te)
            intersection = max(0, inter_end - inter_start)
            
            if intersection > 0:
                # Объединение
                union = (pe - ps) + (te - ts) - intersection
                iou = intersection / union if union > 0 else 0.0
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = idx
        
        if best_match is not None:
            matched_true.add(best_match)
        
        ious.append(best_iou)
    
    return {
        'mean_iou': np.mean(ious) if ious else 0.0,
        'median_iou': np.median(ious) if ious else 0.0,
        'min_iou': np.min(ious) if ious else 0.0,
        'max_iou': np.max(ious) if ious else 0.0,
        'ious': ious
    }

def calculate_mean_time_error(pred_segs, true_segs):
    """
    Рассчитывает среднюю ошибку во времени начала/окончания
    
    Returns:
        dict с ошибками времени
    """
    start_errors = []
    end_errors = []
    matched_true = set()
    
    for ps, pe in pred_segs:
        best_match = None
        best_overlap = 0
        
        for idx, (ts, te) in enumerate(true_segs):
            if idx in matched_true:
                continue
            
            inter_start = max(ps, ts)
            inter_end = min(pe, te)
            overlap = max(0, inter_end - inter_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = (ts, te)
        
        if best_match:
            ts, te = best_match
            start_errors.append(abs(ps - ts))
            end_errors.append(abs(pe - te))
            matched_true.add(true_segs.index((ts, te)))
    
    return {
        'mean_start_error': np.mean(start_errors) if start_errors else 0.0,
        'mean_end_error': np.mean(end_errors) if end_errors else 0.0,
        'median_start_error': np.median(start_errors) if start_errors else 0.0,
        'median_end_error': np.median(end_errors) if end_errors else 0.0,
        'start_errors': start_errors,
        'end_errors': end_errors
    }

def collect_segment_stats(loader, file_items):
    meta = []
    tc = loader.config['data']['target_col']
    for item in tqdm(file_items, desc="Stats"):
        df = loader.load_series(item['file_upload'])
        if df.empty: continue
        for ann in loader.extract_annotations(item):
            seg = df.iloc[ann['start']:ann['end']+1]
            if seg.empty: continue
            d = ann['end'] - ann['start'] + 1
            meta.append({'label': ann['label'], 'duration': d, 'max_speed': seg[tc].max(), 'mean_speed': seg[tc].mean(), 'zero_ratio': np.sum(seg[tc] == 0) / d})
    df_m = pd.DataFrame(meta)
    if df_m.empty: return pd.DataFrame(), []
    res = df_m.groupby('label').agg({'label': 'count', 'duration': ['mean', 'std', 'min', 'max'], 'max_speed': 'mean', 'mean_speed': 'mean', 'zero_ratio': ['mean', 'std']})
    res.columns = ['_'.join(c).strip() for c in res.columns.values]
    return res.rename(columns={'label_count': 'count'}), meta