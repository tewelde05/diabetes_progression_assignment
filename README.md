# Diabetes Progression Modelling

**Author:** Senay Teweldebrhan  
**Student ID:** 9120588  
**Scope:** Machine Learning Assignment, Parts 1–3

## Objective

Predict the quantitative disease-progression score one year after baseline using the scikit-learn Diabetes dataset: 442 patients, ten baseline measurements, and one numerical target. The report evaluates six BMI-only polynomial regressors and eight multivariate candidates using a fixed train–validation–test split. Predictions represent progression-score units, not diabetes diagnoses or clinical risk probabilities.

The main report is [notebooks/Diabetes_Progression_Report.ipynb](notebooks/Diabetes_Progression_Report.ipynb). It contains executed outputs, EDA, model comparisons, equations, prediction examples, parameter counts, and failure analyses. [docs/RUBRIC_COVERAGE.md](docs/RUBRIC_COVERAGE.md) maps the supplied rubric to report sections and source modules.

## Project organization

| Location                                      | Responsibility                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------ |
| `notebooks/Diabetes_Progression_Report.ipynb` | Submission report and analysis orchestration                             |
| `src/data.py`                                 | Dataset loading, quality checks, and patient splitting                   |
| `src/models.py`                               | Predeclared pipelines and explicit logistic score adapter                |
| `src/evaluation.py`                           | Metrics, validation selection, coefficients, and diagnostics             |
| `src/experiment.py`                           | Reproducible experiment and result exports                               |
| `src/plots.py`                                | Labelled EDA, fitted-curve, and residual figures                         |
| `src/notebook_runner.py`                      | Batch execution of explicit report code cells                            |
| `tests/test_workflow.py`                      | Numerical, split-integrity, and leakage checks                           |
| `reports/tables/`                             | Full-precision metrics, coefficients, scaling, split indices, and errors |
| `reports/figures/`                            | Six report figures                                                       |
| `requirements.txt`                            | Pinned numerical and notebook-kernel dependencies                        |

## Environment and reproduction

The numerical workflow was verified with **Python 3.12.14**, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, Matplotlib 3.10.8, and scikit-learn 1.8.0. Python 3.12 is the reference environment. The dataset is bundled with scikit-learn; no credentials or external data download are needed after dependency installation.

Example Windows PowerShell environment setup, from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The report can be opened in VS Code or another Jupyter frontend using this environment as its Python kernel. Run cells in order. The `src/` directory must remain alongside `notebooks/` because the report imports its source modules.

The following commands reproduce the experiment exports, execute and populate the report without a notebook server, and run the checks:

```bash
python -m src.experiment
python -m src.notebook_runner
python -m unittest discover -s tests -v
```

The batch report executor runs explicit Python code cells in one shared namespace and embeds tables and PNG figures. It is not an implementation of Jupyter magics, widgets, or the kernel protocol. The supplied report uses explicit Python cells supported by both this executor and Jupyter.

## Experimental design

- Split seed: **42**; **331 training, 44 validation, and 67 test patients**.
- Detailed EDA uses training patients; deterministic integrity checks cover the supplied table.
- No missing values, non-finite values, duplicate records, or constant predictors require cleaning. Extreme measurements are retained without evidence of recording errors.
- All learned scaling is fitted on training data inside pipelines. All ten original features are retained in Part 3.
- Selection minimizes **unrounded validation MAE**. Exact ties follow the declared candidate order. R² and MAPE are also reported for every candidate on training and validation.
- Model settings are declared before scoring. Selected models remain fitted on training data alone. Test results do not change the choices.

| Part | Models                                                                                   |
| ---- | ---------------------------------------------------------------------------------------- |
| 2    | Six BMI-only ordinary-least-squares polynomials, degrees 0–5                             |
| 3    | Multivariate degree-2 and degree-3 Ridge polynomials, fixed α = 1                        |
| 3    | Decision-tree regressors, maximum depths 3 and 5, minimum leaf size 5                    |
| 3    | Standardized kNN regressors, k = 15 and k = 5, uniform weights                           |
| 3    | Standardized logistic classifiers, C = 0.1 and C = 1, with the explicit adaptation below |

Ridge regularization is used for the multivariate polynomial models because correlated predictors and powers of the binary sex code produce related or redundant expanded terms. Degree 0 has one fitted constant equal to the training target mean. Each degree-d univariate model has d + 1 regression coefficients, including exactly one intercept.

## Logistic-regression interpretation

The original outcome is quantitative, whereas logistic regression is a classifier. The supplied instructions do not define a binary event. The logistic experiments explicitly define the exploratory event **progression score ≥ training median (142)** and report accuracy, precision, recall, F1, ROC AUC, log-loss, and the selected classifier's confusion matrix. This event is not a clinical high-risk definition.

For the required original-target R², MAE, and MAPE comparison, an experimental decoder maps the predicted event probability to a score:

`predicted_score = (1 - probability) * training_mean_below + probability * training_mean_above`

The threshold and both class means use training outcomes only. Classification and score-prediction results are distinguished in the report. This adaptation discards within-band outcome detail and is not ordinary logistic regression on continuous targets. It is a documented methodological assumption because the assignment combines a numerical target with a classifier requirement.

## Results

The BMI-only degree-5 model and the multivariate degree-2 Ridge model are selected by validation MAE. Each multivariate family also contributes its own validation-selected model to the test comparison.

| Validation-selected model         | Test R² | Test MAE | Test MAPE |
| --------------------------------- | ------: | -------: | --------: |
| BMI degree 5                      |   0.267 |    51.75 |    43.03% |
| Degree-2 multivariate Ridge       |   0.433 |    43.67 |    36.56% |
| Tree depth 3                      |   0.449 |    45.80 |    39.73% |
| kNN k = 5                         |   0.471 |    42.21 |    34.65% |
| Logistic C = 1 with score decoder |   0.445 |    46.00 |    41.08% |

The observed kNN test advantage does not replace the model selected on validation data. The multivariate cubic and deeper tree fit training data more closely but generalize less well on validation. The selected multivariate polynomial reduces average error relative to BMI alone, yet its largest test error is approximately 166.63 score units. The report examines underprediction of high outcomes, overprediction of low outcomes, extrapolation, and decoder limitations. These educational results do not establish clinical screening performance.

## Validation

The report's **24 Python code cells** execute successfully and generate **six figures**. The **ten automated checks** cover disjoint and aligned patient partitions, source preservation, constant-model behaviour, training-only normalization, independence from test values during selection, equation/pipeline agreement, original-score percentage metrics, logistic training statistics and decoding, coefficient counts, and cached test predictions. Validation details are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

## Sources

- [Diabetes dataset description](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) and [loader API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).
- [Model-evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), and [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- Course materials: Performance Metrics for Classification; K-Nearest Neighbors Workshop; Logistic Regression & Log-Loss — Pass/Fail Workshop. Specific connections and source filenames appear in the report bibliography.

## AI Assistance

- ChatGPT by OpenAI was used to assist.
