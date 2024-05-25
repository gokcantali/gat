import os
from math import floor, log10

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gat.data_models import Metrics


def shorten_large_numbers(x):
    if x == 0:
        return 0
    elif x < 0:
        return -round(abs(x), -int(floor(log10(abs(x)))) + 2)
    else:
        return round(x, -int(floor(log10(abs(x)))) + 2)

def visualize_data(df, file_path):
    # Correlation matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", annot_kws={"size": 8})
    plt.title("Correlation Matrix", fontsize=16)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "correlation_matrix.png"), bbox_inches="tight")
    plt.clf()

    # Skewness
    skewness = df.skew().sort_values()
    plt.figure(figsize=(14, 6))
    sns.barplot(x=skewness.index, y=skewness.values, palette="viridis", hue=skewness.index, dodge=False, legend=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.yscale("log")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Log Skewness", fontsize=12)
    plt.title("Skewness of Features", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "skewness.png"), bbox_inches="tight")
    plt.clf()

    # Kurtosis
    kurtosis = df.kurt().sort_values()
    plt.figure(figsize=(14, 6))
    sns.barplot(x=kurtosis.index, y=kurtosis.values, palette="magma", hue=kurtosis.index, dodge=False, legend=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.yscale("log")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Log Kurtosis", fontsize=12)
    plt.title("Kurtosis of Features", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "kurtosis.png"), bbox_inches="tight")
    plt.clf()

    # Descriptive statistics table
    desc = df.describe().transpose()
    if "timestamp" in desc.index:
        desc.loc["timestamp"] = desc.loc["timestamp"].apply(shorten_large_numbers)
    desc = desc.apply(np.vectorize(str))
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=desc.values, colLabels=desc.columns, rowLabels=desc.index, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.5, 1.5)
    plt.title("Descriptive Statistics", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "descriptive_statistics_table.png"), bbox_inches="tight")
    plt.clf()

    # Unique values with logarithmic scale
    nunique = df.nunique().sort_values()
    plt.figure(figsize=(14, 6))
    sns.barplot(x=nunique.index, y=nunique.values, palette="coolwarm", hue=nunique.index, dodge=False, legend=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.yscale("log")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Log Number of Unique Values", fontsize=12)
    plt.title("Number of Unique Values in Each Column", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "nunique.png"), bbox_inches="tight")
    plt.clf()

def get_data_insights(df, file_path):
    with open(file_path, "w") as f:
        f.write("\nData info:\n")
        f.write("----------------------------------------------\n")
        df.info(buf=f)

        f.write("\nData describe:\n")
        f.write("----------------------------------------------\n")
        f.write(df.describe().to_string())

        f.write("\nData nunique:\n")
        f.write("----------------------------------------------\n")
        f.write(df.nunique().to_string())

        f.write("\nData correlation:\n")
        f.write("----------------------------------------------\n")
        f.write(df.corr().to_string())

        f.write("\nData skew:\n")
        f.write("----------------------------------------------\n")
        f.write(df.skew().to_string())

        f.write("\nData kurt:\n")
        f.write("----------------------------------------------\n")
        f.write(df.kurt().to_string())

def print_epoch_stats(
    epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy,
    train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, cm
):
    print(f"Epoch {epoch}, Time: {epoch_time:.2f}s")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1-Score: {train_f1:.4f}")
    print(f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1-Score: {val_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("--------------------------------------------------")

def plot_metrics(metrics: Metrics):
    _, ax = plt.subplots(figsize=(12, 8))

#    ax.plot(metrics.epoch_values, metrics.train_loss, marker="o", linestyle="-",
#            color="b", label="Train Loss", markersize=5, linewidth=2)
#    ax.plot(metrics.epoch_values, metrics.val_loss, marker="s", linestyle="--",
#            color="b", label="Val Loss", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.train_accuracy, marker="o", linestyle="-",
            color="r", label="Train Accuracy", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.val_accuracy, marker="s", linestyle="--",
            color="r", label="Val Accuracy", markersize=5, linewidth=2)
#    ax.plot(metrics.epoch_values, metrics.train_precision, marker="o", linestyle="-",
#            color="g", label="Train Precision", markersize=5, linewidth=2)
#    ax.plot(metrics.epoch_values, metrics.val_precision, marker="s", linestyle="--",
#            color="g", label="Val Precision", markersize=5, linewidth=2)
#    ax.plot(metrics.epoch_values, metrics.train_recall, marker="o", linestyle="-",
#            color="y", label="Train Recall", markersize=5, linewidth=2)
#    ax.plot(metrics.epoch_values, metrics.val_recall, marker="s", linestyle="--",
#            color="y", label="Val Recall", markersize=5, linewidth=2)
#    ax.plot(metrics.epoch_values, metrics.train_f1, marker="o", linestyle="-",
#            color="m", label="Train F1-Score", markersize=5, linewidth=2)
#    ax.plot(metrics.epoch_values, metrics.val_f1, marker="s", linestyle="--",
#            color="m", label="Val F1-Score", markersize=5, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Metrics", fontsize=16)
    ax.set_title("Accuracy over Epoch", fontsize=18)

    ax.grid(True, linestyle="--", linewidth=0.7)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig("./results/accuracy_over_epochs.png")
    plt.close()
