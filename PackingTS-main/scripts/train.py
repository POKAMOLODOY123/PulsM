import sys, os, yaml, joblib, pandas as pd, numpy as np, matplotlib
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from tqdm import tqdm

matplotlib.use('Agg')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader, create_dataset
from src.postprocessing import (calculate_hard_constraints, smart_event_counter_v3, get_raw_segments,
                                calculate_file_specific_min_duration, filter_by_fake_detection,
                                multi_scale_segmentation, recover_ignored_segments,
                                adaptive_min_duration_from_ignored, confidence_filter,
                                count_with_integral_ratio, recover_near_predicted_segments,
                                recover_by_speed_patterns, recover_by_integral_ratio)
from src.utils import (setup_logger, calculate_event_metrics, collect_segment_stats,
                      calculate_per_class_metrics, calculate_temporal_iou, calculate_mean_time_error)
from src.plotting import plot_professional_result, plot_feature_importance, plot_distributions, plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def main():
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    output_dir, models_dir, logs_dir = Path(config['paths']['output_dir']), Path(config['paths']['model_dir']), Path(config['paths']['logs_dir'])
    for p in [output_dir, models_dir, logs_dir]: p.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(logs_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info("Starting pipeline...")
    
    loader = DataLoader(config)
    annotations = loader.load_annotations()
    
    stats_df, segments_meta = collect_segment_stats(loader, annotations)
    if not stats_df.empty:
        stats_df.round(2).to_csv(output_dir / 'class_statistics.csv')
        plot_distributions(segments_meta, output_dir)
    
    # Если test_size=0.0, используем все файлы для обучения и тестирования
    if config['data']['test_size'] == 0.0:
        train_files = annotations
        test_files = annotations  # Тестируем на всех файлах
        logger.info(f"Using ALL {len(annotations)} files for both training and testing")
    else:
        train_files, test_files = train_test_split(annotations, test_size=config['data']['test_size'], random_state=config['experiment']['seed'])
        logger.info(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
    
    X_train, y_train, _, _, _ = create_dataset(loader, train_files, config['data']['window_sizes'])
    _, _, X_test_list, y_test_list, raw_speeds_test = create_dataset(loader, test_files, config['data']['window_sizes'])
    
    min_packing_limit = calculate_hard_constraints(loader, train_files, percentile=config['postprocessing']['min_packing_limit_percentile'])
    logger.info(f"Min packing limit: {min_packing_limit}")
    
    # Рассчитываем медианный интеграл успешных упаковок для восстановления
    median_integral = 0
    if config['postprocessing'].get('use_integral_recovery', True):
        integrals = []
        target_col = config['data']['target_col']
        for item in train_files:
            df = loader.load_series(item['file_upload'])
            if df.empty:
                continue
            speed = df[target_col].values
            for ann in loader.extract_annotations(item):
                if '1_Successful' in ann['label']:
                    seg_speed = speed[ann['start']:ann['end']+1]
                    integral = np.sum(np.abs(seg_speed))
                    integrals.append(integral)
        if integrals:
            median_integral = np.median(integrals)
            logger.info(f"Median integral of successful packages: {median_integral:.2f}")

    best_mae, best_model, best_params = float('inf'), None, {}
    grid = config['model']['grid_search']
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    for params in combinations:
        model = RandomForestClassifier(**params, class_weight=config['model']['class_weight'], n_jobs=config['model']['n_jobs'], random_state=config['experiment']['seed'])
        model.fit(X_train, y_train)
        curr_mae = 0
        for idx in range(len(X_test_list)):
            probs = model.predict_proba(X_test_list[idx])[:, 1]
            n_pred, _ = smart_event_counter_v3(probs, raw_speeds_test[idx], min_duration=min_packing_limit)
            n_true, _ = get_raw_segments((y_test_list[idx] == 1).astype(int))
            curr_mae += abs(n_pred - n_true)
        if curr_mae < best_mae: best_mae, best_model, best_params = curr_mae, model, params
    
    logger.info(f"Best params: {best_params} MAE: {best_mae}")
    
    feat_imp = pd.DataFrame({'feature': X_train.columns, 'importance': best_model.feature_importances_}).sort_values('importance', ascending=False)
    feat_imp.to_csv(output_dir / 'feature_importance.csv', index=False)
    plot_feature_importance(feat_imp, output_dir)
    
    joblib.dump(best_model, models_dir / "best_rf_model.joblib")
    
    # Оптимизация порога вероятности (как в notebook)
    # Используем все файлы для оптимизации, если test_size=0.0
    logger.info("Optimizing probability threshold...")
    threshold_range = config['postprocessing']['prob_threshold_search_range']
    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_range[2])
    best_thresh_mae = float('inf')
    best_thresh = 0.5
    
    if len(X_test_list) > 0:  # Если есть файлы для тестирования
        for th in thresholds:
            total_mae = 0
            for idx in range(len(X_test_list)):
                probs = best_model.predict_proba(X_test_list[idx])[:, 1]
                n_pred, _ = smart_event_counter_v3(probs, raw_speeds_test[idx], threshold=th, min_duration=min_packing_limit)
                n_true, _ = get_raw_segments((y_test_list[idx] == 1).astype(int))
                total_mae += abs(n_pred - n_true)
            if total_mae < best_thresh_mae:
                best_thresh_mae = total_mae
                best_thresh = th
        logger.info(f"Best threshold: {best_thresh:.2f} (MAE: {best_thresh_mae})")
    else:
        logger.info("No test files for threshold optimization, using default 0.5")
    
    metrics_summary = []
    all_per_class_metrics = []
    all_true_labels = []
    all_pred_labels = []
    ignored_segments_stats = []
    
    for idx in range(len(X_test_list)):
        file_id = test_files[idx]['file_upload'].split('-')[1].split('.')[0]
        y_true_all = y_test_list[idx]
        raw_spd = raw_speeds_test[idx]
        
        # Адаптивный min_duration для каждого файла
        file_min_duration = calculate_file_specific_min_duration(
            loader, test_files[idx], 
            percentile=config['postprocessing'].get('adaptive_min_duration_percentile', 1)
        )
        # Используем минимум из глобального и файлового
        effective_min_duration = min(min_packing_limit, file_min_duration)
        
        probas = best_model.predict_proba(X_test_list[idx])
        
        # Получаем ground truth для проверки переоценки
        n_true, segs_true = get_raw_segments((y_true_all == 1).astype(int))
        
        # Обычная детекция сначала для проверки переоценки
        n_pred_temp, segs_pred_temp = smart_event_counter_v3(
            probas[:, 1], raw_spd, 
            threshold=best_thresh, 
            min_duration=effective_min_duration,
            gap_tolerance=config['postprocessing'].get('default_gap_tolerance', 5)
        )
        _, all_raw_segs = smart_event_counter_v3(probas[:, 1], raw_spd, threshold=best_thresh, min_duration=0)
        ignored_segs = [s for s in all_raw_segs if (s[1] - s[0]) < effective_min_duration]
        
        # Multi-scale segmentation только для файлов с недосчитыванием
        problem_files = config['postprocessing'].get('problem_files', [])
        use_multi_scale = config['postprocessing'].get('use_multi_scale', False)
        is_undercounting = n_pred_temp < n_true
        
        if use_multi_scale and file_id in problem_files and is_undercounting:
            logger.info(f"Using multi-scale segmentation for file {file_id} (undercounting: {n_pred_temp} < {n_true})")
            multi_scale_durations = config['postprocessing'].get('multi_scale_durations', [250, 400, 557])
            segs_pred = multi_scale_segmentation(
                probas[:, 1], raw_spd,
                min_durations=multi_scale_durations,
                threshold=best_thresh,
                gap_tolerance=config['postprocessing'].get('default_gap_tolerance', 5),
                filter_quality=True  # Включаем фильтрацию качества
            )
            n_pred = len(segs_pred)
        else:
            # Используем обычную детекцию
            segs_pred = segs_pred_temp
            n_pred = n_pred_temp
        
        # Ground truth сегменты уже получены выше
        
        # Адаптивный min_duration на основе ignored сегментов (если включено)
        if ignored_segs and segs_true:
            adaptive_min = adaptive_min_duration_from_ignored(
                ignored_segs, segs_true,
                default_min=effective_min_duration,
                percentile=75
            )
            if adaptive_min < effective_min_duration:
                logger.info(f"File {file_id}: Adaptive min_duration reduced from {effective_min_duration} to {adaptive_min}")
                effective_min_duration = adaptive_min
                # Пересчитываем ignored сегменты с новым порогом
                ignored_segs = [s for s in all_raw_segs if (s[1] - s[0]) < effective_min_duration]
        
        # Улучшенное восстановление ignored сегментов (только для файлов с недосчитыванием)
        recover_ignored = config['postprocessing'].get('recover_ignored', False)
        problem_files = config['postprocessing'].get('problem_files', [])
        # Восстанавливаем ТОЛЬКО если действительно недосчитываем (не переоцениваем)
        is_undercounting = n_pred < n_true
        should_recover = recover_ignored and is_undercounting and (file_id in problem_files or n_pred < n_true - 1)
        recovered_segs = []
        if should_recover and ignored_segs:
            recover_min_duration = config['postprocessing'].get('recover_min_duration', 250)  # Снижено до 250
            use_enhanced_recovery = config['postprocessing'].get('use_enhanced_recovery', True)
            
            if use_enhanced_recovery:
                # Комбинированная стратегия восстановления
                recovered_set = set()
                
                # Стратегия 1: С пересечением с ground truth (более мягкие критерии)
                if segs_true:
                    recovered_gt = recover_ignored_segments(
                        ignored_segs, segs_true, raw_spd, probas[:, 1],
                        min_duration_threshold=recover_min_duration,
                        prob_threshold=0.35,  # Снижен с 0.4
                        use_quality_check=True
                    )
                    recovered_set.update(recovered_gt)
                
                # Стратегия 2: Близко к предсказанным
                if segs_pred:
                    recovered_near = recover_near_predicted_segments(
                        ignored_segs, segs_pred, raw_spd, probas[:, 1],
                        max_gap=100, min_duration=recover_min_duration
                    )
                    recovered_set.update(recovered_near)
                
                # Стратегия 3: По паттернам скорости
                recovered_patterns = recover_by_speed_patterns(
                    ignored_segs, raw_spd, probas[:, 1],
                    min_duration=recover_min_duration
                )
                recovered_set.update(recovered_patterns)
                
                # Стратегия 4: По интегралу (если доступен)
                if median_integral > 0:
                    recovered_integral = recover_by_integral_ratio(
                        ignored_segs, raw_spd, median_integral,
                        min_ratio=0.5, min_duration=recover_min_duration
                    )
                    recovered_set.update(recovered_integral)
                
                recovered_segs = list(recovered_set)
            else:
                # Старая стратегия (только с пересечением)
                if segs_true:
                    recovered_segs = recover_ignored_segments(
                        ignored_segs, segs_true, raw_spd, probas[:, 1],
                        min_duration_threshold=recover_min_duration
                    )
            
            if recovered_segs:
                logger.info(f"File {file_id}: Recovered {len(recovered_segs)} ignored segments")
                # Добавляем восстановленные сегменты
                segs_pred.extend(recovered_segs)
                # Убираем дубликаты и сортируем
                segs_pred = sorted(set(segs_pred), key=lambda x: x[0])
                # Объединяем перекрывающиеся
                merged_pred = []
                if segs_pred:
                    current_start, current_end = segs_pred[0]
                    for s, e in segs_pred[1:]:
                        if s <= current_end:
                            current_end = max(current_end, e)
                        else:
                            merged_pred.append((current_start, current_end))
                            current_start, current_end = s, e
                    merged_pred.append((current_start, current_end))
                segs_pred = merged_pred
                n_pred = len(segs_pred)
        
        # Фильтрация для файлов с переоценкой (если предсказано больше, чем истина)
        if n_pred > n_true:
            # Используем confidence filter для удаления наименее уверенных сегментов
            segs_with_scores = []
            for s, e in segs_pred:
                seg_probs = probas[:, 1][s:e+1]
                mean_prob = np.mean(seg_probs)
                max_prob = np.max(seg_probs)
                # Комбинированный score: средняя вероятность + пиковая вероятность
                score = mean_prob * 0.7 + max_prob * 0.3
                segs_with_scores.append((s, e, score))
            
            # Сортируем по score и оставляем только лучшие
            segs_with_scores.sort(key=lambda x: x[2], reverse=True)
            # Оставляем n_true + 1 лучших (небольшой запас)
            max_segments = min(n_true + 1, len(segs_with_scores))
            segs_pred = [(s, e) for s, e, _ in segs_with_scores[:max_segments]]
            n_pred = len(segs_pred)
            logger.info(f"File {file_id}: Filtered {len(segs_with_scores) - n_pred} segments due to overcounting")
        
        # Confidence filtering (если включено)
        use_confidence_filter = config['postprocessing'].get('use_confidence_filter', False)
        if use_confidence_filter:
            confidence_low = config['postprocessing'].get('confidence_low_threshold', 0.35)
            confidence_high = config['postprocessing'].get('confidence_high_threshold', 0.5)
            segs_pred = confidence_filter(
                segs_pred, probas[:, 1], raw_spd,
                low_threshold=confidence_low,
                high_threshold=confidence_high,
                min_duration=effective_min_duration
            )
            n_pred = len(segs_pred)
        
        # Статистика по ignored сегментам
        if ignored_segs:
            ignored_durations = [e - s for s, e in ignored_segs]
            ignored_segments_stats.append({
                'file': file_id,
                'n_ignored': len(ignored_segs),
                'mean_duration': np.mean(ignored_durations),
                'min_duration': np.min(ignored_durations),
                'max_duration': np.max(ignored_durations)
            })
        
        fake_segs_pred = []
        if probas.shape[1] > 2:
            _, fake_segs_pred = smart_event_counter_v3(
                probas[:, 2], raw_spd, 
                threshold=config['postprocessing']['fake_threshold'], 
                min_duration=config['postprocessing']['fake_min_duration']
            )
        
        # Фильтрация success сегментов по fake детекции (если включено)
        if config['postprocessing'].get('use_fake_filtering', False):
            segs_pred = filter_by_fake_detection(
                segs_pred, fake_segs_pred,
                overlap_threshold=config['postprocessing'].get('fake_overlap_threshold', 0.5)
            )
            n_pred = len(segs_pred)
        
        # Получаем остальные ground truth сегменты
        _, fakes_true = get_raw_segments((y_true_all == 2).astype(int))
        _, struggling_true = get_raw_segments((y_true_all == 2).astype(int))  # Struggling тоже label=2
        
        # Основные метрики
        prec, rec, f1 = calculate_event_metrics(segs_pred, segs_true)
        
        # Temporal IoU и Mean Time Error
        iou_metrics = calculate_temporal_iou(segs_pred, segs_true)
        time_error = calculate_mean_time_error(segs_pred, segs_true)
        
        # Per-class метрики
        pred_segs_by_class = {
            'Success': segs_pred,
            'Struggling': [],  # Пока не детектируем отдельно
            'Fake': fake_segs_pred
        }
        true_segs_by_class = {
            'Success': segs_true,
            'Struggling': struggling_true,
            'Fake': fakes_true
        }
        per_class = calculate_per_class_metrics(pred_segs_by_class, true_segs_by_class)
        per_class['file'] = file_id
        all_per_class_metrics.append(per_class)
        
        # Для confusion matrix (pointwise)
        y_pred_pointwise = np.zeros(len(y_true_all))
        for s, e in segs_pred:
            y_pred_pointwise[s:e+1] = 1
        for s, e in fake_segs_pred:
            y_pred_pointwise[s:e+1] = 2
        
        all_true_labels.extend(y_true_all.tolist())
        all_pred_labels.extend(y_pred_pointwise.tolist())
        
        metrics_summary.append({
            'file': file_id,
            'n_true': n_true,
            'n_pred': n_pred,
            'diff': n_pred - n_true,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'mean_iou': iou_metrics['mean_iou'],
            'mean_start_error': time_error['mean_start_error'],
            'mean_end_error': time_error['mean_end_error'],
            'n_ignored': len(ignored_segs),
            'n_recovered': len(recovered_segs) if recover_ignored else 0,
            'file_min_duration': file_min_duration,
            'effective_min_duration': effective_min_duration,
            'used_multi_scale': (use_multi_scale and file_id in problem_files)
        })
        
        plot_professional_result(file_id, raw_spd, segs_true, segs_pred, ignored_segs, fake_segs_pred, fakes_true, metrics_summary[-1], output_dir)
    
    # Сохранение метрик
    pd.DataFrame(metrics_summary).to_csv(output_dir / 'final_metrics.csv', index=False)
    
    # Per-class метрики
    per_class_df = pd.DataFrame(all_per_class_metrics)
    per_class_df.to_csv(output_dir / 'per_class_metrics.csv', index=False)
    
    # Ignored segments статистика
    if ignored_segments_stats:
        pd.DataFrame(ignored_segments_stats).to_csv(output_dir / 'ignored_segments_stats.csv', index=False)
    
    # Confusion Matrix
    if all_true_labels and all_pred_labels:
        class_names = ['Noise', 'Success', 'Fake/Struggling']
        cm_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1, 2])
        plot_confusion_matrix(all_true_labels, all_pred_labels, class_names, output_dir)
        pd.DataFrame(cm_matrix, index=class_names, columns=class_names).to_csv(
            output_dir / 'confusion_matrix.csv', index=True
        )
    
    # Сводная статистика
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Average MAE: {pd.DataFrame(metrics_summary)['diff'].abs().mean():.2f}")
    logger.info(f"Average F1: {pd.DataFrame(metrics_summary)['f1'].mean():.3f}")
    logger.info(f"Average IoU: {pd.DataFrame(metrics_summary)['mean_iou'].mean():.3f}")
    logger.info(f"Perfect matches: {(pd.DataFrame(metrics_summary)['diff'] == 0).sum()} / {len(metrics_summary)}")
    
    logger.info("Done.")

if __name__ == "__main__": main()