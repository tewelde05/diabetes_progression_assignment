# Validation Record

**Date:** 2026-10-19  
**Reference Python:** 3.12.14  
**Random seed:** 42

## Executed workflow

- The independent `python -m src.experiment` entry point fitted all 14 declared candidates and exported the experiment tables.
- All 24 explicit report code cells executed sequentially in one shared Python namespace through `src.notebook_runner`, producing six embedded figures with no cell errors.
- Notebook tables, figures, coefficient exports, and conclusions use the same fixed model specifications and patient split.
- The ten built-in `unittest` checks passed.
- Report figures were visually inspected for labelled axes, readable tables/legends, subset scope, and consistency with reported metrics.

## Numerical and leakage checks

1. Patient subsets are disjoint, cover all 442 rows, and preserve feature–target alignment.
2. Data cleaning preserves the supplied dataset; missing-value corruption raises a clear error.
3. The degree-0 predictor equals the training response mean for different BMI inputs.
4. BMI normalization matches training-only mean and population standard deviation.
5. Replacing test values does not alter training/validation model comparisons.
6. The full-precision BMI equation reproduces the fitted pipeline's predictions.
7. Logistic labels, threshold, class means, and score decoding use training outcomes and valid probabilities.
8. MAPE percentages are calculated against original numerical outcomes.
9. Candidate counts and fitted polynomial coefficient counts match the experiment design.
10. Cached selected-model test predictions are finite and agree with the fitted estimators.

The BMI equation's two-decimal coefficients differ from the full-precision pipeline by at most approximately 1.337 score units on a 500-point training-range grid. The selected multivariate polynomial's displayed rounded coefficients differ by approximately 0.079 score units on training patients. Full-precision coefficients and preprocessing constants are retained in the exports. These numerical display checks are not clinical tolerances.

The batch execution check is distinct from validation through a Jupyter kernel or frontend. The report uses standard Python cells and standard notebook outputs, and the pinned `ipykernel` dependency supports interactive notebook use.
