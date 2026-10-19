"""Dataset loading, integrity checks, and reproducible patient-level partitions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = 'target'


@dataclass(frozen=True)
class DataSplit:
    """Aligned predictor and target objects for three disjoint patient subsets."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_dataset() -> tuple[pd.DataFrame, str]:
    """Return supplied, unscaled measurements and the official description."""
    dataset = load_diabetes(as_frame=True, scaled=False)
    return dataset.frame.copy(), dataset.DESCR


def quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe completeness and finite numeric values without fitting transforms."""
    return pd.DataFrame({
        'dtype': frame.dtypes.astype(str),
        'missing': frame.isna().sum(),
        'non_finite': (~np.isfinite(frame)).sum(),
        'unique_values': frame.nunique(),
    })


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the bundled dataset; preserve valid unusual patient measurements."""
    if TARGET not in frame or not frame.index.is_unique:
        raise ValueError('A target and unique patient row identifiers are required.')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Missing or infinite values require an explicit cleaning policy.')
    if frame.duplicated().any():
        raise ValueError('Duplicate records require provenance review before splitting.')
    if not frame['sex'].isin([1, 2]).all():
        raise ValueError('Unexpected sex codes in the unscaled source data.')
    return frame.copy()


def split_dataset(frame: pd.DataFrame, seed: int = SEED) -> DataSplit:
    """Allocate 75/10/15 percent, with unavoidable whole-patient rounding."""
    train, holdout = train_test_split(frame.index, test_size=0.25, random_state=seed)
    validation, test = train_test_split(holdout, test_size=0.60, random_state=seed)
    X, y = frame.drop(columns=TARGET), frame[TARGET]
    return DataSplit(
        X.loc[train].copy(), X.loc[validation].copy(), X.loc[test].copy(),
        y.loc[train].copy(), y.loc[validation].copy(), y.loc[test].copy(),
    )


def training_frame(split: DataSplit) -> pd.DataFrame:
    """Combine training predictors and responses for descriptive EDA only."""
    return split.X_train.join(split.y_train)


def split_summary(split: DataSplit) -> pd.DataFrame:
    """Show requested and realized patient allocations."""
    sizes = [len(split.y_train), len(split.y_val), len(split.y_test)]
    return pd.DataFrame({
        'Subset': ['Training', 'Validation', 'Test'],
        'Patients': sizes,
        'Requested (%)': [75, 10, 15],
        'Actual (%)': np.array(sizes) / sum(sizes) * 100,
    })


def iqr_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Flag extreme continuous training measurements without deleting observations."""
    values = train.drop(columns=[TARGET, 'sex'])
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return pd.DataFrame({
        'Lower fence': low, 'Upper fence': high,
        'Flagged values': ((values < low) | (values > high)).sum(),
    })
