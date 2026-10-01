import pandas as pd
import warnings
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from typing import Tuple

def active_region_split(manifest_df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group by active_region_id so no AR appears in multiple splits."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Split into train_val and test
    train_val_idx, test_idx = next(gss.split(manifest_df, groups=manifest_df['active_region_id']))
    train_val_df = manifest_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = manifest_df.iloc[test_idx].reset_index(drop=True)
    
    # Split train_val into train and val
    val_ratio = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['active_region_id']))
    
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    
    return train_df, val_df, test_df

def temporal_split(manifest_df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by timestamp and split chronologically."""
    df = manifest_df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    
    test_idx = int(n * (1.0 - test_size))
    val_idx = int(n * (1.0 - test_size - val_size))
    
    train_df = df.iloc[:val_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx:test_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx:].reset_index(drop=True)
    
    return train_df, val_df, test_df

def random_split(manifest_df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simple random split with a warning about data leakage."""
    warnings.warn("Using random_split on spatio-temporal data can cause data leakage. Consider active_region_split or temporal_split.")
    
    train_val_df, test_df = train_test_split(manifest_df, test_size=test_size, random_state=random_state)
    
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=random_state)
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
