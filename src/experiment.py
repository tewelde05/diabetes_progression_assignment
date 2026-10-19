"""Run the predeclared experiments and export independently reproducible results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

from .data import DataSplit, clean_dataset, load_dataset, split_dataset, split_summary
from .evaluation import (
    fit_and_compare, logistic_metrics, parameter_summary, polynomial_coefficients,
    prediction_diagnostics, select_candidate, selected_test_results,
)
from .models import LogisticScoreAdapter, bmi_candidates, multivariate_candidates


def run_experiment(split: DataSplit) -> dict:
    """Fit/select all candidates before computing any new test predictions."""
    bmi = bmi_candidates()
    multivariate = multivariate_candidates(split.X_train.columns.tolist())
    bmi_table = fit_and_compare(bmi, split)
    multi_table = fit_and_compare(multivariate, split)
    bmi_winner = select_candidate(bmi_table, bmi)
    multi_winner = select_candidate(multi_table, multivariate)
    family_winners = [select_candidate(multi_table, multivariate, family)
                      for family in dict.fromkeys(c.family for c in multivariate)]
    selected = [bmi_winner, *family_winners]
    test_table, predictions = selected_test_results(selected, split)
    classification_rows = []
    for candidate in multivariate:
        if not isinstance(candidate.estimator, LogisticScoreAdapter):
            continue
        for label, X, y in [('Train', split.X_train, split.y_train),
                            ('Validation', split.X_val, split.y_val)]:
            classification_rows.append({'Model': candidate.name, 'Subset': label,
                                        **logistic_metrics(candidate, X, y)})
    selected_logistic = next(c for c in family_winners if isinstance(c.estimator, LogisticScoreAdapter))
    classification_rows.append({'Model': selected_logistic.name, 'Subset': 'Test',
                                **logistic_metrics(selected_logistic, split.X_test, split.y_test)})
    diagnostic_tables, diagnostic_bands = {}, {}
    for candidate in selected:
        details, bands = prediction_diagnostics(split.y_test, predictions[candidate.name], split.X_test)
        diagnostic_tables[candidate.name] = details
        diagnostic_bands[candidate.name] = bands
    profile = split.X_train.median().to_frame().T
    profile.loc[:, 'bmi'] = 25.0
    profile.loc[:, 'sex'] = int(split.X_train['sex'].mode().iloc[0])
    example_rows = []
    for candidate in multivariate:
        example_rows.append({'Model': candidate.name,
                             'Predicted score': float(candidate.estimator.predict(profile[candidate.columns])[0])})
    return {
        'bmi_candidates': bmi, 'multivariate_candidates': multivariate,
        'bmi_table': bmi_table, 'multivariate_table': multi_table,
        'bmi_winner': bmi_winner, 'multivariate_winner': multi_winner,
        'family_winners': family_winners, 'test_table': test_table,
        'test_predictions': predictions, 'classification_table': pd.DataFrame(classification_rows),
        'parameters': parameter_summary([*bmi, *multivariate]),
        'diagnostics': diagnostic_tables, 'diagnostic_bands': diagnostic_bands,
        'example_profile': profile, 'example_predictions': pd.DataFrame(example_rows),
    }


def safe_name(name: str) -> str:
    """Convert fixed report labels to portable export filenames."""
    return ''.join(character if character.isalnum() else '_' for character in name).strip('_')


def export_results(result: dict, split: DataSplit, directory: Path) -> None:
    """Save tables, exact coefficients, preprocessing statistics, and tree rules."""
    directory.mkdir(parents=True, exist_ok=True)
    for key in ['bmi_table', 'multivariate_table', 'test_table', 'classification_table',
                'parameters', 'example_profile', 'example_predictions']:
        result[key].to_csv(directory / f'{key}.csv', index=False, float_format='%.17g')
    split_summary(split).to_csv(directory / 'split_summary.csv', index=False)
    allocations = pd.concat([
        pd.DataFrame({'patient_row': X.index, 'subset': subset})
        for subset, X in [('train', split.X_train), ('validation', split.X_val), ('test', split.X_test)]
    ]).sort_values('patient_row')
    allocations.to_csv(directory / 'patient_split.csv', index=False)
    for candidate in [*result['bmi_candidates'], *result['multivariate_candidates']]:
        model, stem = candidate.estimator, safe_name(candidate.name)
        if isinstance(model, Pipeline) and 'poly' in model.named_steps:
            polynomial_coefficients(candidate).to_csv(directory / f'{stem}_coefficients.csv', index=False, float_format='%.17g')
            scaler = model.named_steps['scale']
            pd.DataFrame({'Feature': candidate.columns, 'Mean': scaler.mean_, 'Scale': scaler.scale_}).to_csv(
                directory / f'{stem}_input_scaling.csv', index=False, float_format='%.17g')
            if 'term_scale' in model.named_steps:
                term_scaler = model.named_steps['term_scale']
                terms = model.named_steps['poly'].get_feature_names_out([f'z_{c}' for c in candidate.columns])
                pd.DataFrame({'Term': terms, 'Mean': term_scaler.mean_, 'Scale': term_scaler.scale_}).to_csv(
                    directory / f'{stem}_term_scaling.csv', index=False, float_format='%.17g')
        elif isinstance(model, DecisionTreeRegressor):
            (directory / f'{stem}_rules.txt').write_text(
                export_text(model, feature_names=candidate.columns, decimals=12), encoding='utf-8')
            tree = model.tree_
            pd.DataFrame({'Node': np.arange(tree.node_count), 'Left': tree.children_left,
                          'Right': tree.children_right, 'Feature_index': tree.feature,
                          'Threshold': tree.threshold, 'Value': tree.value.ravel()}).to_csv(
                directory / f'{stem}_nodes.csv', index=False, float_format='%.17g')
        elif isinstance(model, LogisticScoreAdapter):
            logistic = model.classifier_.named_steps['logistic']
            pd.DataFrame({'Term': ['intercept', *candidate.columns],
                          'Coefficient': np.r_[logistic.intercept_, logistic.coef_.ravel()]}).to_csv(
                directory / f'{stem}_logit_coefficients.csv', index=False, float_format='%.17g')
            scale = model.classifier_.named_steps['scale']
            pd.DataFrame({'Feature': candidate.columns, 'Mean': scale.mean_, 'Scale': scale.scale_}).to_csv(
                directory / f'{stem}_input_scaling.csv', index=False, float_format='%.17g')
    for name, table in result['diagnostics'].items():
        table.to_csv(directory / f'{safe_name(name)}_test_errors.csv', index_label='patient_row', float_format='%.17g')
        result['diagnostic_bands'][name].to_csv(directory / f'{safe_name(name)}_error_bands.csv', float_format='%.17g')
    summary = {
        'seed': 42, 'split_counts': [len(split.y_train), len(split.y_val), len(split.y_test)],
        'selection_rule': 'Minimum validation MAE; exact ties follow declared candidate order.',
        'bmi_winner': result['bmi_winner'].name,
        'multivariate_winner': result['multivariate_winner'].name,
        'family_winners': [c.name for c in result['family_winners']],
        'logistic_adaptation': 'Training-median binary labels with training-class-mean probability decoding.',
    }
    (directory / 'experiment_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main() -> None:
    """Run from the project root with python -m src.experiment."""
    frame, _ = load_dataset()
    split = split_dataset(clean_dataset(frame))
    result = run_experiment(split)
    root = Path(__file__).resolve().parents[1]
    export_results(result, split, root / 'reports' / 'tables')
    print(result['multivariate_table'].drop(columns='Order').round(3).to_string(index=False))
    print('\nValidation-selected models: test results')
    print(result['test_table'].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
