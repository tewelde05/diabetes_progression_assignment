# Rubric Coverage

## Part 1 — 5 points

| Requirement | Report section | Evidence |
|---|---|---|
| 1. Get the data | 1 | Official loader, options, dataset dimensions, feature dictionary |
| 2. Frame the problem | 2 | One-year numerical outcome, screening motivation, three workshop connections, evaluation design |
| 3. EDA — 4 points | 3 | Training statistics, ten feature scatter plots, histograms, sex-code counts, correlation matrix, and interpretations |
| 4. Clean if needed — 1 point | 4 | Missing/non-finite/duplicate/constant checks, IQR flags, justified preservation of source data |
| 5. Split 75/10/15 | 3 and 5 | 331/44/67 rows, seed 42, patient-disjoint and target-alignment checks |

## Part 2 — 10 points

| Requirement | Report section | Evidence |
|---|---|---|
| 6. Six BMI polynomials — 1.5 points | 6 | Degrees 0–5; `bmi_candidates()` in `src/models.py` |
| 7. Train/validation metric table — 3 points | 7 | Separate row per model; R², MAE, MAPE on both subsets |
| 8. Best model | 8 | Degree 5 selected by minimum validation MAE |
| 9. Test metrics — 1 point | 9 | Selected-model R², MAE, and MAPE |
| 10. Data and fitted curve — 1 point | 10 | Three labelled subset panels with the same training-fitted curve |
| 11. Equation — 1 point | 11 | Two-decimal standardized-BMI equation; coordinate definition and precision check |
| 12. `model.predict()` example — 0.5 point | 12 | BMI 25.0, expected score 137.19 |
| 13. Trainable parameters — 1 point | 13 | Named polynomial terms and 1–6 coefficient counts |
| 14. Conclusion/failures/limitations — 1 point | 14 | Error bands, largest failures, residual plot, extrapolation, and workshop implications |

## Part 3 — 5 points

| Repeated requirement | Report section | Evidence |
|---|---|---|
| Feature policy and eight candidates | 15 | All ten features; degrees 2/3, depths 3/5, k 15/5, C 0.1/1 |
| Steps 6–8: fit, compare, select | 15–16 | Eight-row train/validation table, family winners, overall validation-MAE winner |
| Step 9: selected-model tests | 17 | Four family-selected models evaluated with original-score R²/MAE/MAPE |
| Step 10: plots | Not required by assignment | No multivariate fitted-surface plots; residual diagnostics are supplemental |
| Step 11: equations/rules | 18 | Complete selected polynomial coefficient table and scalings; tree rule export; kNN averaging rule; logistic equation and decoder |
| Step 12: expected progression | 19 | Explicit complete hypothetical profile with BMI 25; predictions for all eight candidates |
| Step 13: fitted quantities | 19 | 66/286 polynomial coefficients, tree responses/thresholds, kNN stored samples, 11 logistic coefficients plus adapter statistics |
| Step 14: conclusions/failures | 20–21 | Logistic classification metrics, false negatives, regression error bands, worst cases, overfitting, limitations, workshop synthesis |

### Methodological qualification

The rubric specifies logistic regression while retaining a numerical target. Sections 2 and 15–20 explicitly state the training-median binary-event assumption, separate classification metrics from score metrics, and define a training-only probability-to-score adapter. The report does not claim that this threshold is clinically validated or that logistic regression directly predicts a continuous response. Multivariate polynomial regression uses fixed Ridge regularization, explicitly stated and justified by the training EDA.
