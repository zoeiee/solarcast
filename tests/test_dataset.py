import pytest
import pandas as pd
import numpy as np

# Mocking dataset structures
class SolarDataset:
    def __init__(self, manifest_df, data_dir):
        self.df = manifest_df
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'image': np.zeros((1, 128, 128)),
            'label': row['has_flare'],
            'id': row['id']
        }

def get_splits(df):
    train = df.iloc[:2]
    val = df.iloc[2:3]
    test = df.iloc[3:]
    return train, val, test

def test_solar_dataset_length(sample_manifest_df, temp_data_dir):
    ds = SolarDataset(sample_manifest_df, temp_data_dir)
    assert len(ds) == len(sample_manifest_df)

def test_solar_dataset_item(sample_manifest_df, temp_data_dir):
    ds = SolarDataset(sample_manifest_df, temp_data_dir)
    item = ds[0]
    assert 'image' in item
    assert 'label' in item
    assert 'id' in item
    assert item['image'].shape == (1, 128, 128)

def test_active_region_split(sample_manifest_df):
    train, val, test = get_splits(sample_manifest_df)
    train_ids = set(train['id'])
    val_ids = set(val['id'])
    test_ids = set(test['id'])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

def test_temporal_split(sample_manifest_df):
    train, val, test = get_splits(sample_manifest_df.sort_values('timestamp'))
    train_max_time = train['timestamp'].max()
    val_min_time = val['timestamp'].min()
    assert train_max_time <= val_min_time

def test_split_ratios(sample_manifest_df):
    train, val, test = get_splits(sample_manifest_df)
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1

def test_no_data_leakage(sample_manifest_df):
    train, val, test = get_splits(sample_manifest_df)
    train_paths = set(train['file_path'])
    val_paths = set(val['file_path'])
    test_paths = set(test['file_path'])
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
