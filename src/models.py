"""Predeclared model specifications and the explicitly adapted logistic models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from .data import SEED


@dataclass
class Candidate:
    """An estimator, its report label, and the columns supplied to it."""

    name: str
    family: str
    estimator: BaseEstimator
    columns: list[str]
    order: int


class LogisticScoreAdapter(RegressorMixin, BaseEstimator):
    """Fit a median-defined binary classifier and decode probabilities to scores.

    This is an experimental classifier-based score predictor, not ordinary
    logistic regression on continuous outcomes. Threshold and decoding means
    are estimated exclusively from the response passed to fit().
    """

    def __init__(self, C: float = 1.0, random_state: int = SEED):
        self.C = C
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn a training-only label definition, class means, and classifier."""
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError('A finite one-dimensional response is required.')
        self.threshold_ = float(np.median(values))
        labels = (values >= self.threshold_).astype(int)
        if len(np.unique(labels)) != 2:
            raise ValueError('Both median-defined classes must be represented.')
        self.class_means_ = np.array([values[labels == k].mean() for k in (0, 1)])
        self.classifier_ = Pipeline([
            ('scale', StandardScaler()),
            ('logistic', LogisticRegression(
                C=self.C, solver='lbfgs', max_iter=5000,
                random_state=self.random_state,
            )),
        ])
        self.classifier_.fit(X, labels)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probabilities for the two exploratory progression bands."""
        check_is_fitted(self, ['classifier_', 'class_means_', 'threshold_'])
        return self.classifier_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Decode binary probabilities to original progression-score units."""
        return self.predict_proba(X) @ self.class_means_


def bmi_candidates() -> list[Candidate]:
    """Specify all six univariate least-squares polynomial regressors."""
    return [Candidate(
        f'BMI degree {degree}', 'BMI polynomial', Pipeline([
            ('scale', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
            ('regression', LinearRegression(fit_intercept=False)),
        ]), ['bmi'], degree,
    ) for degree in range(6)]


def multivariate_candidates(columns: list[str]) -> list[Candidate]:
    """Specify eight candidates before validation or test results are computed.

    Ridge regularizes correlated polynomial terms. kNN candidates differ only
    in k; trees differ only in depth; logistic candidates differ only in C.
    """
    candidates = []
    for degree in (2, 3):
        candidates.append(Candidate(
            f'Polynomial degree {degree} (Ridge)', 'Multivariate polynomial',
            Pipeline([
                ('scale', StandardScaler()),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('term_scale', StandardScaler()),
                ('regression', Ridge(alpha=1.0, solver='svd')),
            ]), columns.copy(), len(candidates),
        ))
    for depth in (3, 5):
        candidates.append(Candidate(
            f'Tree depth {depth}', 'Decision tree',
            DecisionTreeRegressor(max_depth=depth, min_samples_leaf=5, random_state=SEED),
            columns.copy(), len(candidates),
        ))
    for k in (15, 5):
        candidates.append(Candidate(
            f'kNN k={k}', 'kNN', Pipeline([
                ('scale', StandardScaler()),
                ('regression', KNeighborsRegressor(n_neighbors=k, weights='uniform', p=2)),
            ]), columns.copy(), len(candidates),
        ))
    for C in (0.1, 1.0):
        candidates.append(Candidate(
            f'Logistic C={C:g} (score adapter)', 'Logistic score adapter',
            LogisticScoreAdapter(C=C), columns.copy(), len(candidates),
        ))
    return candidates
