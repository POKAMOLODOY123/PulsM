import json
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config['paths']['data_dir'])
        self.series_dir = Path(config['paths']['series_dir'])
        
    def load_annotations(self):
        path = self.data_dir / self.config['paths']['annotations_file']
        with open(path, 'r') as f:
            return json.load(f)

    def load_series(self, filename):
        clean_name = filename.split('-')[1].split('.')[0] + ".txt"
        filepath = self.series_dir / clean_name
        try:
            df = pd.read_csv(filepath, sep=';', parse_dates=['Time'])
            col = self.config['data']['target_col']
            df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except FileNotFoundError:
            return pd.DataFrame()

    def extract_annotations(self, item):
        results = []
        if 'annotations' in item and len(item['annotations']) > 0:
            ann = item['annotations'][0]
            if 'result' in ann:
                for r in ann['result']:
                    if r['type'] == 'timeserieslabels':
                        results.append({
                            'start': r['value']['start'],
                            'end': r['value']['end'],
                            'label': r['value']['timeserieslabels'][0]
                        })
        return results

def create_dataset(loader, file_items, window_sizes):
    X_list = []
    y_list = []
    raw_speeds = []
    
    target_col = loader.config['data']['target_col']
    
    for item in file_items:
        df = loader.load_series(item['file_upload'])
        if df.empty: continue
            
        speed = df[target_col].values
        y = np.zeros(len(speed), dtype=int)
        
        annotations = loader.extract_annotations(item)
        for ann in annotations:
            label_code = 0
            if '1_Successful' in ann['label']: label_code = 1
            elif '3_Fake' in ann['label'] or '2_Struggling' in ann['label']: label_code = 2
            
            if label_code > 0:
                y[ann['start']:ann['end']+1] = label_code
        
        features = pd.DataFrame({'speed': speed})
        features['speed_lag_50'] = features['speed'].shift(50).fillna(0)
        
        for w in window_sizes:
            roll = features['speed'].rolling(w, center=True, min_periods=1)
            features[f'roll_mean_{w}'] = roll.mean()
            features[f'roll_max_{w}'] = roll.max()
            features[f'roll_std_{w}'] = roll.std()
            features[f'zero_ratio_{w}'] = (features['speed'] == 0).rolling(w, center=True, min_periods=1).mean()
            
        features = features.fillna(0)
        X_list.append(features)
        y_list.append(y)
        raw_speeds.append(speed)

    if not X_list:
        raise ValueError("No data loaded")

    X_full = pd.concat(X_list, ignore_index=True)
    y_full = np.concatenate(y_list)
    return X_full, y_full, X_list, y_list, raw_speeds