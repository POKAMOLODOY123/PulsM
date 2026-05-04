import numpy as np
from scipy.ndimage import label as nd_label

def get_raw_segments(mask):
    labeled, num = nd_label(mask)
    segments = []
    for i in range(1, num + 1):
        where = np.where(labeled == i)[0]
        if len(where) > 0:
            segments.append((where[0], where[-1]))
    return len(segments), segments

def smart_event_counter_v3(probs_pack, raw_speed, threshold=0.5, min_duration=100, gap_tolerance=10):
    mask = (probs_pack >= threshold).astype(int)
    is_stopped = (raw_speed <= 2).astype(int)
    stopped_labels, num_stops = nd_label(is_stopped)
    refined_mask = mask.copy()
    for i in range(1, num_stops + 1):
        if np.sum(stopped_labels == i) >= gap_tolerance:
            refined_mask[stopped_labels == i] = 0
    labeled_array, num_features = nd_label(refined_mask)
    valid_segments = []
    for i in range(1, num_features + 1):
        where = np.where(labeled_array == i)[0]
        if len(where) >= min_duration:
            valid_segments.append((where[0], where[-1]))
    return len(valid_segments), valid_segments

def calculate_hard_constraints(loader, file_list, percentile=1):
    durations = []
    for item in file_list:
        anns = loader.extract_annotations(item)
        for ann in anns:
            if '1_Successful' in ann['label']:
                durations.append(ann['end'] - ann['start'])
    return int(np.percentile(durations, percentile)) if durations else 100

def calculate_file_specific_min_duration(loader, file_item, percentile=1):
    """
    Рассчитывает min_packing_limit для конкретного файла
    
    Args:
        loader: DataLoader
        file_item: элемент из annotations
        percentile: перцентиль для расчета
    
    Returns:
        min_duration для этого файла
    """
    durations = []
    anns = loader.extract_annotations(file_item)
    for ann in anns:
        if '1_Successful' in ann['label']:
            durations.append(ann['end'] - ann['start'])
    
    if durations:
        return int(np.percentile(durations, percentile))
    # Если нет успешных упаковок в этом файле, используем глобальный
    return 100

def filter_by_fake_detection(success_segs, fake_segs, overlap_threshold=0.5):
    """
    Фильтрует success сегменты, которые пересекаются с fake сегментами
    
    Args:
        success_segs: список success сегментов [(start, end), ...]
        fake_segs: список fake сегментов [(start, end), ...]
        overlap_threshold: минимальный процент пересечения для исключения
    
    Returns:
        отфильтрованный список success сегментов
    """
    if not fake_segs:
        return success_segs
    
    filtered_segs = []
    
    for s_start, s_end in success_segs:
        should_exclude = False
        
        for f_start, f_end in fake_segs:
            # Пересечение
            inter_start = max(s_start, f_start)
            inter_end = min(s_end, f_end)
            intersection = max(0, inter_end - inter_start)
            
            if intersection > 0:
                # Процент пересечения относительно success сегмента
                overlap_ratio = intersection / (s_end - s_start) if (s_end - s_start) > 0 else 0
                
                if overlap_ratio >= overlap_threshold:
                    should_exclude = True
                    break
        
        if not should_exclude:
            filtered_segs.append((s_start, s_end))
    
    return filtered_segs

def multi_scale_segmentation(probs_pack, raw_speed, min_durations=[300, 557, 800], 
                             threshold=0.5, gap_tolerance=5, filter_quality=True):
    """
    Ищет сегменты на разных масштабах длительности с фильтрацией качества
    
    Args:
        probs_pack: вероятности упаковки
        raw_speed: массив скоростей
        min_durations: список минимальных длительностей для поиска
        threshold: порог вероятности
        gap_tolerance: допуск на разрывы
        filter_quality: фильтровать сегменты по качеству
    
    Returns:
        объединенный список сегментов
    """
    all_segments = []
    
    for min_dur in min_durations:
        _, segs = smart_event_counter_v3(probs_pack, raw_speed, threshold=threshold, 
                                       min_duration=min_dur, gap_tolerance=gap_tolerance)
        all_segments.extend(segs)
    
    # Объединяем близкие сегменты и убираем дубликаты
    if not all_segments:
        return []
    
    # Сортируем по началу
    all_segments = sorted(all_segments, key=lambda x: x[0])
    
    # Объединяем перекрывающиеся сегменты
    merged = []
    current_start, current_end = all_segments[0]
    
    for start, end in all_segments[1:]:
        if start <= current_end:  # Пересекаются или близки
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged.append((current_start, current_end))
    
    # Фильтрация по качеству (если включено)
    if filter_quality:
        filtered = []
        for s, e in merged:
            seg_probs = probs_pack[s:e+1]
            seg_speed = raw_speed[s:e+1]
            mean_prob = np.mean(seg_probs)
            max_prob = np.max(seg_probs)
            max_speed = np.max(seg_speed) if len(seg_speed) > 0 else 0
            mean_speed = np.mean(seg_speed) if len(seg_speed) > 0 else 0
            zero_ratio = np.sum(seg_speed == 0) / (e - s) if (e - s) > 0 else 1.0
            
            # Более строгие критерии для multi-scale сегментов
            if (mean_prob > 0.4 and  # Достаточная средняя вероятность
                max_prob > 0.5 and  # Высокая пиковая вероятность
                max_speed > 8 and  # Достаточная максимальная скорость
                mean_speed > 3 and  # Достаточная средняя скорость
                zero_ratio < 0.6):  # Не слишком много нулей
                filtered.append((s, e))
        
        return filtered
    
    return merged

def recover_ignored_segments(ignored_segs, true_segs, raw_speed, probs, 
                            min_duration_threshold=300, prob_threshold=0.35,
                            use_quality_check=True):
    """
    Улучшенное восстановление ignored сегментов
    
    Args:
        ignored_segs: список ignored сегментов
        true_segs: список реальных сегментов (ground truth)
        raw_speed: массив скоростей
        probs: вероятности упаковки
        min_duration_threshold: минимальная длительность для восстановления
        prob_threshold: порог вероятности (снижен с 0.4 до 0.35)
        use_quality_check: использовать проверку качества без пересечения
    
    Returns:
        список восстановленных сегментов
    """
    recovered = []
    
    for ign_start, ign_end in ignored_segs:
        duration = ign_end - ign_start
        seg_probs = probs[ign_start:ign_end+1]
        seg_speed = raw_speed[ign_start:ign_end+1]
        
        mean_prob = np.mean(seg_probs)
        max_prob = np.max(seg_probs)
        max_speed = np.max(seg_speed) if len(seg_speed) > 0 else 0
        mean_speed = np.mean(seg_speed) if len(seg_speed) > 0 else 0
        zero_ratio = np.sum(seg_speed == 0) / duration if duration > 0 else 1.0
        
        # Стратегия 1: С пересечением с ground truth (более мягкие критерии)
        found_in_gt = False
        for true_start, true_end in true_segs:
            inter_start = max(ign_start, true_start)
            inter_end = min(ign_end, true_end)
            
            if inter_end > inter_start:  # Есть пересечение
                # Более мягкие критерии
                if (mean_prob > prob_threshold and 
                    max_speed > 5 and
                    duration >= min_duration_threshold):
                    recovered.append((ign_start, ign_end))
                    found_in_gt = True
                    break
        
        # Стратегия 2: Без пересечения, но с проверкой качества (более строгие критерии)
        if not found_in_gt and use_quality_check:
            # Проверяем качественные признаки упаковки (ужесточенные критерии)
            if (mean_prob > 0.4 and  # Повышен с 0.35
                max_prob > 0.5 and  # Повышен с 0.45
                max_speed > 8 and  # Повышен с 5
                mean_speed > 3 and  # Повышен с 2
                zero_ratio < 0.6 and  # Снижен с 0.7
                duration >= min_duration_threshold):
                recovered.append((ign_start, ign_end))
    
    return recovered

def recover_near_predicted_segments(ignored_segs, pred_segs, raw_speed, probs,
                                   max_gap=100, min_duration=250):
    """
    Восстанавливает ignored сегменты, которые близки к предсказанным
    
    Args:
        ignored_segs: список ignored сегментов
        pred_segs: список предсказанных сегментов
        raw_speed: массив скоростей
        probs: вероятности упаковки
        max_gap: максимальный разрыв для объединения
        min_duration: минимальная длительность
    
    Returns:
        список восстановленных сегментов
    """
    recovered = []
    
    for ign_start, ign_end in ignored_segs:
        duration = ign_end - ign_start
        
        if duration < min_duration:
            continue
        
        # Проверяем близость к предсказанным сегментам
        for pred_start, pred_end in pred_segs:
            # Расстояние до предсказанного сегмента
            gap_before = pred_start - ign_end if ign_end < pred_start else 0
            gap_after = ign_start - pred_end if ign_start > pred_end else 0
            gap = max(gap_before, gap_after)
            
            # Если близко и сегмент качественный (ужесточенные критерии)
            if 0 < gap <= max_gap:
                seg_probs = probs[ign_start:ign_end+1]
                seg_speed = raw_speed[ign_start:ign_end+1]
                mean_prob = np.mean(seg_probs)
                max_prob = np.max(seg_probs)
                max_speed = np.max(seg_speed) if len(seg_speed) > 0 else 0
                mean_speed = np.mean(seg_speed) if len(seg_speed) > 0 else 0
                
                if (mean_prob > 0.4 and  # Повышен с 0.35
                    max_prob > 0.5 and  # Добавлена проверка пиковой вероятности
                    max_speed > 8 and  # Повышен с 5
                    mean_speed > 3 and  # Добавлена проверка средней скорости
                    duration >= min_duration):
                    recovered.append((ign_start, ign_end))
                    break
    
    return recovered

def recover_by_speed_patterns(ignored_segs, raw_speed, probs,
                              min_duration=250):
    """
    Восстанавливает ignored сегменты по паттернам скорости
    
    Args:
        ignored_segs: список ignored сегментов
        raw_speed: массив скоростей
        probs: вероятности упаковки
        min_duration: минимальная длительность
    
    Returns:
        список восстановленных сегментов
    """
    recovered = []
    
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return recovered
    
    for ign_start, ign_end in ignored_segs:
        duration = ign_end - ign_start
        
        if duration < min_duration:
            continue
        
        seg_speed = raw_speed[ign_start:ign_end+1]
        seg_probs = probs[ign_start:ign_end+1]
        
        # Анализ паттернов
        max_speed = np.max(seg_speed)
        mean_speed = np.mean(seg_speed)
        
        # Интеграл скорости
        integral = np.sum(np.abs(seg_speed))
        
        # Количество пиков (локальных максимумов)
        if max_speed > 0:
            peaks, _ = find_peaks(seg_speed, height=max_speed * 0.3)
            n_peaks = len(peaks)
        else:
            n_peaks = 0
        
        # Производная (изменение скорости)
        if len(seg_speed) > 1:
            derivative = np.diff(seg_speed)
            max_derivative = np.max(np.abs(derivative))
        else:
            max_derivative = 0
        
        # Критерии для восстановления
        mean_prob = np.mean(seg_probs)
        
        if (n_peaks >= 3 and  # Повышен с 2 до 3
            integral > duration * 15 and  # Повышен с 10 до 15
            max_derivative > 8 and  # Повышен с 5 до 8
            mean_prob > 0.4 and  # Повышен с 0.3 до 0.4
            max_speed > 8):  # Повышен с 5 до 8
            recovered.append((ign_start, ign_end))
    
    return recovered

def recover_by_integral_ratio(ignored_segs, raw_speed, median_integral,
                              min_ratio=0.6, min_duration=250):
    """
    Восстанавливает ignored сегменты по отношению интеграла к медианному
    
    Args:
        ignored_segs: список ignored сегментов
        raw_speed: массив скоростей
        median_integral: медианный интеграл успешных упаковок
        min_ratio: минимальное отношение интеграла (повышен с 0.5 до 0.6)
        min_duration: минимальная длительность
    
    Returns:
        список восстановленных сегментов
    """
    recovered = []
    
    if median_integral <= 0:
        return recovered
    
    for ign_start, ign_end in ignored_segs:
        duration = ign_end - ign_start
        
        if duration < min_duration:
            continue
        
        seg_speed = raw_speed[ign_start:ign_end+1]
        integral = np.sum(np.abs(seg_speed))
        ratio = integral / median_integral
        
        # Если интеграл достаточен относительно медианного (ужесточенный критерий)
        if ratio >= min_ratio:
            recovered.append((ign_start, ign_end))
    
    return recovered

def adaptive_min_duration_from_ignored(ignored_segs, true_segs, 
                                       default_min=557, percentile=75):
    """
    Определяет min_duration на основе ignored сегментов
    
    Args:
        ignored_segs: список ignored сегментов
        true_segs: список реальных сегментов
        default_min: значение по умолчанию
        percentile: перцентиль для анализа
    
    Returns:
        адаптивный min_duration
    """
    if not ignored_segs or not true_segs:
        return default_min
    
    # Средняя длительность реальных упаковок
    true_durations = [e - s for s, e in true_segs]
    if not true_durations:
        return default_min
    
    mean_true_duration = np.mean(true_durations)
    
    # Перцентиль длительности ignored сегментов
    ignored_durations = [e - s for s, e in ignored_segs]
    if ignored_durations:
        ignored_percentile = np.percentile(ignored_durations, percentile)
        
        # Если ignored сегменты близки к реальным - снижаем порог
        if ignored_percentile > mean_true_duration * 0.5:
            # Используем более мягкий порог
            return int(mean_true_duration * 0.6)
    
    return default_min

def confidence_filter(segs, probs, raw_speed, low_threshold=0.35, 
                     high_threshold=0.5, min_duration=300):
    """
    Фильтрует сегменты с низкой уверенностью
    
    Args:
        segs: список сегментов
        probs: вероятности упаковки
        raw_speed: массив скоростей
        low_threshold: нижний порог вероятности
        high_threshold: верхний порог вероятности
        min_duration: минимальная длительность
    
    Returns:
        отфильтрованный список сегментов
    """
    filtered = []
    
    for s, e in segs:
        seg_probs = probs[s:e+1]
        seg_speed = raw_speed[s:e+1]
        mean_prob = np.mean(seg_probs)
        duration = e - s
        
        # Если вероятность в пограничной зоне
        if low_threshold <= mean_prob < high_threshold:
            # Дополнительные проверки
            max_speed = np.max(seg_speed) if len(seg_speed) > 0 else 0
            zero_ratio = np.sum(seg_speed == 0) / duration if duration > 0 else 1.0
            
            # Должны быть признаки реальной упаковки
            if (duration >= min_duration and max_speed > 10 and zero_ratio < 0.5):
                filtered.append((s, e))
        elif mean_prob >= high_threshold:
            # Высокая уверенность - принимаем
            filtered.append((s, e))
    
    return filtered

def count_with_integral_ratio(segs, raw_speed, median_integral, 
                             ratio_threshold=1.6):
    """
    Считает упаковки с учетом интеграла (как у коллеги)
    
    Args:
        segs: список сегментов
        raw_speed: массив скоростей
        median_integral: медианный интеграл успешных упаковок
        ratio_threshold: порог отношения интеграла
    
    Returns:
        общее количество упаковок
    """
    if median_integral <= 0:
        return len(segs)
    
    total_count = 0
    
    for s, e in segs:
        seg_speed = raw_speed[s:e+1]
        integral = np.sum(np.abs(seg_speed))
        ratio = integral / median_integral
        
        if ratio >= ratio_threshold:
            # Округляем отношение - может быть несколько упаковок
            count = int(np.round(ratio))
            total_count += max(1, count)  # Минимум 1
        else:
            total_count += 1
    
    return total_count