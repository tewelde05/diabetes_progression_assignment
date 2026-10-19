"""Report figures with labelled axes and explicit patient-subset scope."""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import DataSplit, TARGET
from .models import Candidate


def configure_plots() -> None:
    plt.rcParams.update({
        'figure.dpi': 110, 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold',
    })


def save_figure(figure, destination: Path) -> None:
    """Render completely in memory before replacing a figure export."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=110, bbox_inches='tight', facecolor='white')
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(destination)


def plot_histograms(train: pd.DataFrame):
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for ax, column in zip(axes.flat, train.columns):
        if column == 'sex':
            counts = train[column].value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts, color='#2563eb')
        else:
            ax.hist(train[column], bins=20, color='#2563eb', edgecolor='white', alpha=0.85)
        ax.set(title=column, xlabel=column, ylabel='Patients')
        ax.grid(axis='y', alpha=0.15)
    for ax in axes.flat[len(train.columns):]:
        ax.set_visible(False)
    fig.suptitle('Training data: feature and progression-score distributions', fontsize=15)
    return fig


def plot_scatter(train: pd.DataFrame):
    features = train.drop(columns=TARGET).columns
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    for ax, column in zip(axes.flat, features):
        ax.scatter(train[column], train[TARGET], s=16, alpha=0.5, color='#0f766e')
        correlation = train[column].corr(train[TARGET])
        ax.set(title=f'{column}: r = {correlation:.2f}', xlabel=column, ylabel='Progression score')
        ax.grid(alpha=0.15)
        if column == 'sex':
            ax.set_xticks([1, 2])
    fig.suptitle('Training patients: baseline features versus one-year progression', fontsize=15)
    return fig


def plot_correlations(train: pd.DataFrame):
    matrix = train.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix)), labels=matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(matrix)), labels=matrix.columns)
    for row in range(len(matrix)):
        for col in range(len(matrix)):
            value = matrix.iloc[row, col]
            ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(value) > 0.65 else '#172033')
    ax.set_title('Training data: Pearson correlation matrix', pad=15)
    fig.colorbar(image, ax=ax, label='Pearson r', shrink=0.85)
    return fig


def plot_bmi_fit(candidate: Candidate, split: DataSplit):
    """Plot the same training-fitted BMI curve against all three patient subsets."""
    subsets = [('Training', split.X_train, split.y_train, '#2563eb'),
               ('Validation', split.X_val, split.y_val, '#b45309'),
               ('Test', split.X_test, split.y_test, '#0f766e')]
    low = min(float(X['bmi'].min()) for _, X, _, _ in subsets)
    high = max(float(X['bmi'].max()) for _, X, _, _ in subsets)
    grid = pd.DataFrame({'bmi': np.linspace(low, high, 500)})
    predicted = candidate.estimator.predict(grid)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, (label, X, y, color) in zip(axes, subsets):
        ax.scatter(X['bmi'], y, s=26, alpha=0.65, color=color, label=f'{label} patients')
        ax.plot(grid['bmi'], predicted, color='#111827', linewidth=2, label=candidate.name)
        for left, right in [(low, split.X_train['bmi'].min()), (split.X_train['bmi'].max(), high)]:
            if right > left:
                ax.axvspan(left, right, color='#9ca3af', alpha=0.25)
        ax.set(title=f'{label}: n = {len(y)}', xlabel='BMI (supplied unscaled values)')
        ax.grid(alpha=0.15)
        ax.legend(fontsize=9, loc='upper left')
    axes[0].set_ylabel('One-year progression score')
    fig.suptitle('Selected BMI model: one fitted curve, three patient subsets', fontsize=15)
    return fig


def plot_residuals(details: pd.DataFrame, label: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].scatter(details['actual'], details['predicted'], color='#0f766e', alpha=0.7)
    low, high = details[['actual', 'predicted']].min().min(), details[['actual', 'predicted']].max().max()
    axes[0].plot([low, high], [low, high], '--', color='#111827')
    axes[0].set(title='Actual versus predicted', xlabel='Actual score', ylabel='Predicted score')
    axes[1].scatter(details['actual'], details['residual'], color='#b45309', alpha=0.7)
    axes[1].axhline(0, linestyle='--', color='#111827')
    axes[1].set(title='Errors across observed scores', xlabel='Actual score', ylabel='Actual minus predicted')
    axes[2].hist(details['residual'], bins=15, color='#64748b', edgecolor='white')
    axes[2].axvline(0, linestyle='--', color='#111827')
    axes[2].set(title='Error distribution', xlabel='Actual minus predicted', ylabel='Patients')
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(f'Test diagnostics: {label}', fontsize=15)
    return fig
