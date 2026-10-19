"""Metric calculation, selection, coefficients, and post-selection diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss, mean_absolute_error,
    mean_absolute_percentage_error, precision_score, r2_score, recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeRegressor

from .data import DataSplit
from .models import Candidate, LogisticScoreAdapter


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Use the original positive progression target for all regression metrics."""
    return {
        'R2': float(r2_score(actual, predicted)),
        'MAE': float(mean_absolute_error(actual, predicted)),
        'MAPE (%)': float(100 * mean_absolute_percentage_error(actual, predicted)),
    }


def fit_and_compare(candidates: list[Candidate], split: DataSplit) -> pd.DataFrame:
    """Fit on training patients and evaluate training/validation only."""
    rows = []
    for candidate in candidates:
        candidate.estimator.fit(split.X_train[candidate.columns], split.y_train)
        row = {'Model': candidate.name, 'Family': candidate.family, 'Order': candidate.order}
        for subset, X, y in [
            ('Train', split.X_train, split.y_train),
            ('Validation', split.X_val, split.y_val),
        ]:
            predicted = candidate.estimator.predict(X[candidate.columns])
            row.update({f'{subset} {key}': value for key, value in regression_metrics(y, predicted).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def select_candidate(table: pd.DataFrame, candidates: list[Candidate], family=None) -> Candidate:
    """Minimize unrounded validation MAE; exact ties follow specification order."""
    eligible = table if family is None else table.loc[table['Family'] == family]
    winner = eligible.sort_values(['Validation MAE', 'Order'], kind='stable').iloc[0]['Model']
    return next(candidate for candidate in candidates if candidate.name == winner)


def selected_test_results(selected: list[Candidate], split: DataSplit):
    """Score selected models once and retain predictions for later diagnostics."""
    rows, predictions = [], {}
    for candidate in selected:
        predicted = candidate.estimator.predict(split.X_test[candidate.columns])
        predictions[candidate.name] = predicted
        rows.append({'Model': candidate.name, 'Family': candidate.family,
                     **regression_metrics(split.y_test, predicted)})
    return pd.DataFrame(rows), predictions


def logistic_metrics(candidate: Candidate, X: pd.DataFrame, y: pd.Series) -> dict:
    """Classify the fixed training-median event, at probability threshold 0.5."""
    estimator = candidate.estimator
    actual = (np.asarray(y) >= estimator.threshold_).astype(int)
    probabilities = estimator.predict_proba(X[candidate.columns])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        'Accuracy': float(accuracy_score(actual, predicted)),
        'Precision': float(precision_score(actual, predicted, zero_division=0)),
        'Recall': float(recall_score(actual, predicted, zero_division=0)),
        'F1': float(f1_score(actual, predicted, zero_division=0)),
        'ROC AUC': float(roc_auc_score(actual, probabilities)),
        'Log-loss': float(log_loss(actual, probabilities, labels=[0, 1])),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


def parameter_summary(candidates: list[Candidate]) -> pd.DataFrame:
    """Report fitted quantities without implying equal capacity across families."""
    rows = []
    for candidate in candidates:
        model = candidate.estimator
        row = {'Model': candidate.name, 'Response coefficients / values': 0,
               'Split thresholds': 0, 'Training records retained': 0, 'Interpretation': ''}
        if isinstance(model, LogisticScoreAdapter):
            logistic = model.classifier_.named_steps['logistic']
            row['Response coefficients / values'] = int(logistic.coef_.size + logistic.intercept_.size)
            row['Interpretation'] = '11 classifier coefficients; plus 2 decoding means and the training-median threshold.'
        elif isinstance(model, DecisionTreeRegressor):
            leaves = model.get_n_leaves()
            row['Response coefficients / values'] = int(leaves)
            row['Split thresholds'] = int(model.tree_.node_count - leaves)
            row['Interpretation'] = 'One mean per leaf; one threshold and a discrete feature choice per internal node.'
        elif 'poly' in model.named_steps:
            regression = model.named_steps['regression']
            count = regression.coef_.size + int(regression.fit_intercept)
            row['Response coefficients / values'] = int(count)
            row['Interpretation'] = 'One coefficient per polynomial term, including exactly one intercept.'
        else:
            row['Training records retained'] = int(model.named_steps['regression'].n_samples_fit_)
            row['Interpretation'] = 'No fitted slope/intercept coefficients; stores training features and responses.'
        rows.append(row)
    return pd.DataFrame(rows)


def polynomial_coefficients(candidate: Candidate) -> pd.DataFrame:
    """Name fitted terms and preserve coefficients independently of display rounding."""
    model = candidate.estimator
    poly = model.named_steps['poly']
    names = poly.get_feature_names_out([f'z_{name}' for name in candidate.columns])
    regression = model.named_steps['regression']
    table = pd.DataFrame({'Term': names, 'Coefficient': regression.coef_})
    if regression.fit_intercept:
        table = pd.concat([pd.DataFrame({'Term': ['intercept'], 'Coefficient': [regression.intercept_]}), table], ignore_index=True)
    return table


def prediction_diagnostics(actual: pd.Series, predicted: np.ndarray, X: pd.DataFrame):
    """Return patient errors and descriptive outcome-band error summaries."""
    details = X.copy()
    details['actual'] = actual
    details['predicted'] = predicted
    details['residual'] = details['actual'] - details['predicted']
    details['absolute_error'] = details['residual'].abs()
    details['APE (%)'] = 100 * details['absolute_error'] / details['actual'].abs()
    details['Observed score band'] = pd.cut(
        actual, [-np.inf, 100, 200, np.inf],
        labels=['Below 100', '100 to below 200', '200 or above'], right=False,
    )
    grouped = details.groupby('Observed score band', observed=True).agg(
        Patients=('actual', 'size'), MAE=('absolute_error', 'mean'),
        Mean_residual=('residual', 'mean'), MAPE_percent=('APE (%)', 'mean'),
    )
    return details, grouped
