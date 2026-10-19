"""Numerical and leakage checks for the complete assignment workflow."""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from src.data import clean_dataset, load_dataset, split_dataset
from src.evaluation import fit_and_compare, regression_metrics
from src.experiment import run_experiment
from src.models import LogisticScoreAdapter, bmi_candidates


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, _ = load_dataset()
        cls.split = split_dataset(clean_dataset(cls.frame))
        cls.result = run_experiment(cls.split)

    def test_split_is_disjoint_complete_and_aligned(self):
        split = self.split
        groups = [set(X.index) for X in [split.X_train, split.X_val, split.X_test]]
        self.assertEqual([len(g) for g in groups], [331, 44, 67])
        self.assertEqual(set.union(*groups), set(self.frame.index))
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(groups[i] & groups[j])
        for X, y in [(split.X_train, split.y_train), (split.X_val, split.y_val), (split.X_test, split.y_test)]:
            self.assertTrue(X.index.equals(y.index))
            self.assertNotIn('target', X)

    def test_cleaning_preserves_source_measurements(self):
        pd.testing.assert_frame_equal(self.frame, clean_dataset(self.frame))
        damaged = self.frame.copy()
        damaged.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            clean_dataset(damaged)

    def test_constant_model_is_training_mean_for_any_bmi(self):
        model = self.result['bmi_candidates'][0].estimator
        predicted = model.predict(pd.DataFrame({'bmi': [18.0, 25.0, 50.0]}))
        np.testing.assert_allclose(predicted, self.split.y_train.mean())

    def test_scaling_is_fitted_on_training_patients(self):
        for candidate in self.result['bmi_candidates']:
            scaler = candidate.estimator.named_steps['scale']
            self.assertAlmostEqual(scaler.mean_[0], self.split.X_train['bmi'].mean())
            self.assertAlmostEqual(scaler.scale_[0], self.split.X_train['bmi'].std(ddof=0))

    def test_validation_selection_does_not_access_test_data(self):
        altered = replace(self.split, X_test=self.split.X_test * 1e12,
                          y_test=self.split.y_test * -100)
        table = fit_and_compare(bmi_candidates(), altered)
        pd.testing.assert_frame_equal(table, self.result['bmi_table'])

    def test_selected_bmi_equation_reproduces_pipeline(self):
        model = self.result['bmi_winner'].estimator
        scaler = model.named_steps['scale']
        X = self.split.X_test[['bmi']]
        z = (X['bmi'].to_numpy() - scaler.mean_[0]) / scaler.scale_[0]
        equation = Polynomial(model.named_steps['regression'].coef_)
        np.testing.assert_allclose(equation(z), model.predict(X), atol=1e-10)

    def test_logistic_decoder_uses_training_statistics(self):
        models = [c.estimator for c in self.result['multivariate_candidates']
                  if isinstance(c.estimator, LogisticScoreAdapter)]
        threshold = self.split.y_train.median()
        labels = self.split.y_train >= threshold
        means = [self.split.y_train[~labels].mean(), self.split.y_train[labels].mean()]
        for model in models:
            self.assertEqual(model.threshold_, threshold)
            np.testing.assert_allclose(model.class_means_, means)
            probabilities = model.predict_proba(self.split.X_test)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1)
            np.testing.assert_allclose(model.predict(self.split.X_test), probabilities @ means)
            self.assertTrue(np.all(model.predict(self.split.X_test) >= min(means)))
            self.assertTrue(np.all(model.predict(self.split.X_test) <= max(means)))

    def test_original_score_mape_is_percent(self):
        scores = regression_metrics([100, 200], [80, 160])
        self.assertAlmostEqual(scores['MAPE (%)'], 20.0)
        self.assertAlmostEqual(scores['MAE'], 30.0)

    def test_candidate_counts_and_parameter_terms(self):
        self.assertEqual(len(self.result['bmi_candidates']), 6)
        self.assertEqual(len(self.result['multivariate_candidates']), 8)
        parameters = self.result['parameters'].set_index('Model')
        self.assertEqual(parameters.loc['Polynomial degree 2 (Ridge)', 'Response coefficients / values'], 66)
        self.assertEqual(parameters.loc['Polynomial degree 3 (Ridge)', 'Response coefficients / values'], 286)
        for degree in range(6):
            self.assertEqual(parameters.loc[f'BMI degree {degree}', 'Response coefficients / values'], degree + 1)

    def test_test_predictions_are_finite_and_match_selected_models(self):
        selected = [self.result['bmi_winner'], *self.result['family_winners']]
        self.assertEqual(len(self.result['test_predictions']), 5)
        for candidate in selected:
            cached = self.result['test_predictions'][candidate.name]
            self.assertEqual(len(cached), 67)
            self.assertTrue(np.isfinite(cached).all())
            np.testing.assert_allclose(cached, candidate.estimator.predict(self.split.X_test[candidate.columns]))


if __name__ == '__main__':
    unittest.main()
