import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize

import config
import utils
from data_loader import create_dataloaders
from model import get_model


def calculate_specificity(cm):
    

    specificities = []

    for i in range(len(cm)):
        
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])

       
        fp = np.sum(cm[:, i]) - cm[i, i]

        if (tn + fp) == 0:
            specificity = 0.0
        else:
            specificity = tn / (tn + fp)

        specificities.append(specificity)

    macro_specificity = np.mean(specificities)

    return specificities, macro_specificity


def evaluate(args):

    device = utils.get_device()
    save_path = config.model_save_path(args.model)

    if not os.path.exists(save_path):
        print(f"[ERROR] No checkpoint found at {save_path}")
        print(f"  Train first: python train.py --model {args.model}")
        return
    ckpt = torch.load(save_path, map_location=device)

    class_names = ckpt.get(
        'class_names',
        config.CLASS_NAMES
    )

    model = get_model(
        args.model,
        num_classes=len(class_names)
    ).to(device)

    model.load_state_dict(
        ckpt['model_state_dict']
    )

    model.eval()

    print(
        f"\nLoaded {args.model} "
        f"— best val_acc={ckpt['val_acc']:.2f}%\n"
    )
 
    _, _, test_loader, _ = create_dataloaders(
        verbose=False
    )

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            logits = model(images)

            probs = F.softmax(
                logits,
                dim=1
            ).cpu().numpy()

            preds = logits.argmax(
                dim=1
            ).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("=" * 65)
    print("CLASSIFICATION REPORT")
    print("=" * 65)

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )

    print(report_str)

    report_path = os.path.join(
        config.OUTPUTS_DIR,
        f'classification_report_{args.model}.csv'
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        report_path
    )

    print(
        f"Classification report saved → {report_path}"
    )

    accuracy = (
        all_preds == all_labels
    ).mean()

    macro_precision = report_dict['macro avg']['precision']
    macro_recall = report_dict['macro avg']['recall']
    macro_f1 = report_dict['macro avg']['f1-score']

    macro_sensitivity = macro_recall

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    per_class_specificity, macro_specificity = calculate_specificity(cm)

    print("=" * 65)
    print("ADDITIONAL METRICS")
    print("=" * 65)

    print(f"Accuracy          : {accuracy * 100:.2f}%")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro Sensitivity : {macro_sensitivity:.4f}")
    print(f"Macro Specificity : {macro_specificity:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    print("\nPer-class Specificity:")

    for cls, spec in zip(
        class_names,
        per_class_specificity
    ):
        print(
            f"  {cls:<15}: {spec:.4f}"
        )

    try:

        auc_macro = roc_auc_score(
            all_labels,
            all_probs,
            multi_class='ovr',
            average='macro'
        )

        print(
            f"\nMacro ROC-AUC     : {auc_macro:.4f}"
        )

    except Exception as e:

        auc_macro = None

        print(
            f"\nAUC-ROC could not be computed: {e}"
        )

    y_bin = label_binarize(
        all_labels,
        classes=list(range(len(class_names)))
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for i, cls in enumerate(class_names):

        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            all_probs[:, i]
        )

        auc_i = roc_auc_score(
            y_bin[:, i],
            all_probs[:, i]
        )

        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f'{cls} (AUC={auc_i:.3f})'
        )

    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        lw=1
    )

    ax.set_xlabel(
        'False Positive Rate'
    )

    ax.set_ylabel(
        'True Positive Rate'
    )

    ax.set_title(
        f'ROC Curve — {args.model}'
    )

    ax.legend(
        loc='lower right'
    )

    ax.grid(alpha=0.2)

    roc_path = os.path.join(
        config.OUTPUTS_DIR,
        f'roc_curve_{args.model}.png'
    )

    fig.savefig(
        roc_path,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(
        f"ROC curve saved → {roc_path}"
    )

    roc_data = {
        'model': args.model,
        'class_names': class_names,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

    roc_data_path = os.path.join(
        config.OUTPUTS_DIR,
        f'roc_data_{args.model}.npz'
    )

    np.savez(
        roc_data_path,
        labels=all_labels,
        probabilities=all_probs
    )

    print(
        f"ROC data saved → {roc_data_path}"
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel(
        'Predicted'
    )

    ax.set_ylabel(
        'True'
    )

    ax.set_title(
        f'Confusion Matrix — {args.model}'
    )

    cm_path = os.path.join(
        config.OUTPUTS_DIR,
        f'confusion_matrix_{args.model}.png'
    )

    fig.savefig(
        cm_path,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(
        f"Confusion matrix saved → {cm_path}"
    )

    summary_path = os.path.join(
        config.OUTPUTS_DIR,
        'results_summary.csv'
    )

    row = {
        'model': args.model,
        'accuracy': f'{accuracy * 100:.2f}',
        'precision': f'{macro_precision:.4f}',
        'recall': f'{macro_recall:.4f}',
        'sensitivity': f'{macro_sensitivity:.4f}',
        'specificity': f'{macro_specificity:.4f}',
        'macro_f1': f'{macro_f1:.4f}',
        'macro_auc': (
            f'{auc_macro:.4f}'
            if auc_macro is not None
            else 'N/A'
        )
    }

    df = pd.DataFrame([row])

    if os.path.exists(summary_path):

        df.to_csv(
            summary_path,
            mode='a',
            header=False,
            index=False
        )

    else:

        df.to_csv(
            summary_path,
            index=False
        )

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    print(
        f"Model             : {args.model}"
    )

    print(
        f"Accuracy           : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision          : {macro_precision:.4f}"
    )

    print(
        f"Recall             : {macro_recall:.4f}"
    )

    print(
        f"Sensitivity        : {macro_sensitivity:.4f}"
    )

    print(
        f"Specificity        : {macro_specificity:.4f}"
    )

    print(
        f"F1 Score           : {macro_f1:.4f}"
    )

    if auc_macro is not None:
        print(
            f"ROC-AUC            : {auc_macro:.4f}"
        )

    print("=" * 65)

    print(
        f"\nSummary appended → {summary_path}"
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluate a trained model on BRISC2025'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='custom_cnn',
        choices=[
            'custom_cnn',
            'resnet50',
            'efficientnet',
            'mobilenetv3'
        ]
    )

    args = parser.parse_args()

    evaluate(args)
