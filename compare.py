import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score

import config


SUMMARY_PATH = os.path.join(
    config.OUTPUTS_DIR,
    'results_summary.csv'
)


def compare():

    if not os.path.exists(SUMMARY_PATH):

        print(
            "[ERROR] results_summary.csv not found."
        )

        print(
            "Run eval.py for each model first."
        )

        return

    df = pd.read_csv(
        SUMMARY_PATH
    )

    df = (
        df.drop_duplicates(
            subset='model',
            keep='last'
        )
        .reset_index(drop=True)
    )

    metrics = [
        'accuracy',
        'precision',
        'recall',
        'sensitivity',
        'specificity',
        'macro_f1',
        'macro_auc'
    ]

    for metric in metrics:

        if metric in df.columns:

            df[metric] = pd.to_numeric(
                df[metric],
                errors='coerce'
            )

    print("\n" + "=" * 95)

    print(
        "MODEL COMPARISON — BRISC2025 TEST SET"
    )

    print("=" * 95)

    display_columns = [
        'model',
        'accuracy',
        'precision',
        'recall',
        'sensitivity',
        'specificity',
        'macro_f1',
        'macro_auc'
    ]

    print(
        df[display_columns].to_string(
            index=False
        )
    )

    print("=" * 95)

    comparison_csv = os.path.join(
        config.OUTPUTS_DIR,
        'model_comparison.csv'
    )

    df[display_columns].to_csv(
        comparison_csv,
        index=False
    )

    print(
        f"\nComparison table saved → {comparison_csv}"
    )

    labels = {
        'accuracy': 'Accuracy (%)',
        'precision': 'Precision',
        'recall': 'Recall',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'macro_f1': 'Macro F1',
        'macro_auc': 'Macro ROC-AUC'
    }

    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(17, 9)
    )

    axes = axes.flatten()

    for idx, metric in enumerate(metrics):

        ax = axes[idx]

        vals = df[metric].astype(float)

        plot_vals = vals.copy()

        if metric == 'accuracy':
            y_values = plot_vals
            text_values = plot_vals
            upper_limit = 100
        else:
            y_values = plot_vals
            text_values = plot_vals
            upper_limit = 1.0

        bars = ax.bar(
            df['model'],
            y_values,
            alpha=0.85,
            edgecolor='white'
        )

        ax.set_title(
            labels[metric]
        )

        if metric == 'accuracy':
            ax.set_ylim(
                0,
                max(100, vals.max() * 1.15)
            )
        else:
            ax.set_ylim(
                0,
                min(1.0, max(1.0, vals.max() * 1.15))
            )

        for bar, val in zip(
            bars,
            text_values
        ):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (
                    0.02
                    if metric != 'accuracy'
                    else 1.0
                ),
                f'{val:.3f}'
                if metric != 'accuracy'
                else f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.tick_params(
            axis='x',
            rotation=20
        )

        ax.grid(
            axis='y',
            alpha=0.2
        )

    axes[-1].axis('off')

    fig.suptitle(
        'Model Comparison — BRISC2025',
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()

    comparison_plot = os.path.join(
        config.OUTPUTS_DIR,
        'model_comparison.png'
    )

    fig.savefig(
        comparison_plot,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(
        f"Comparison chart saved → {comparison_plot}"
    )

    plot_combined_roc(df)


def plot_combined_roc(df):

    class_names = config.CLASS_NAMES

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    models_plotted = 0

    for _, row in df.iterrows():

        model_name = row['model']

        roc_data_path = os.path.join(
            config.OUTPUTS_DIR,
            f'roc_data_{model_name}.npz'
        )

        if not os.path.exists(roc_data_path):

            print(
                f"[WARNING] ROC data not found for "
                f"{model_name}"
            )

            continue

        data = np.load(
            roc_data_path
        )

        all_labels = data['labels']
        all_probs = data['probabilities']

        y_bin = label_binarize(
            all_labels,
            classes=list(
                range(len(class_names))
            )
        )

        # Compute macro ROC curve
        fpr_grid = np.linspace(
            0,
            1,
            1000
        )

        tprs = []

        for i in range(
            len(class_names)
        ):

            fpr, tpr, _ = roc_curve(
                y_bin[:, i],
                all_probs[:, i]
            )

            tpr_interp = np.interp(
                fpr_grid,
                fpr,
                tpr
            )

            tprs.append(
                tpr_interp
            )

        macro_tpr = np.mean(
            tprs,
            axis=0
        )

        auc_value = roc_auc_score(
            all_labels,
            all_probs,
            multi_class='ovr',
            average='macro'
        )

        ax.plot(
            fpr_grid,
            macro_tpr,
            lw=2,
            label=(
                f'{model_name} '
                f'(AUC={auc_value:.3f})'
            )
        )

        models_plotted += 1

    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        lw=1,
        label='Random Classifier'
    )

    ax.set_xlabel(
        'False Positive Rate'
    )

    ax.set_ylabel(
        'True Positive Rate'
    )

    ax.set_title(
        'ROC-AUC Comparison — BRISC2025'
    )

    ax.legend(
        loc='lower right'
    )

    ax.grid(
        alpha=0.2
    )

    if models_plotted == 0:

        print(
            "[WARNING] No ROC data available."
        )

        plt.close(fig)

        return

    roc_comparison_path = os.path.join(
        config.OUTPUTS_DIR,
        'combined_roc_comparison.png'
    )

    fig.savefig(
        roc_comparison_path,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(
        f"Combined ROC comparison saved → "
        f"{roc_comparison_path}"
    )


if __name__ == '__main__':

    compare()
