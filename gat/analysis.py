import matplotlib.pyplot as plt
import seaborn as sns
import os

from gat.data_models import Metrics


def shorten_large_numbers(x):
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    elif x >= 1e9:
        return f"{x/1e9:.2f}B"
    elif x >= 1e6:
        return f"{x/1e6:.2f}M"
    elif x >= 1e3:
        return f"{x/1e3:.2f}K"
    else:
        return f"{x:.2f}"

def visualize_data(df, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Descriptions of plots
    descriptions = {
        "correlation_matrix": "This plot shows the correlation matrix of the dataset. It illustrates the pairwise correlation coefficients between features, with values ranging from -1 to 1. A high positive value indicates a strong positive correlation, while a high negative value indicates a strong negative correlation.",
        "skewness": "The skewness plot displays the skewness of each feature in the dataset on a logarithmic scale. Skewness measures the asymmetry of the data distribution. A skewness close to zero indicates a symmetrical distribution, while a positive or negative skewness indicates an asymmetrical distribution.",
        "kurtosis": "The kurtosis plot shows the kurtosis of each feature in the dataset on a logarithmic scale. Kurtosis measures the 'tailedness' of the data distribution. A high kurtosis indicates heavy tails and the presence of outliers, while a low kurtosis indicates light tails.",
        "descriptive_statistics_table": "This table provides the descriptive statistics for each feature in the dataset. It includes the count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum values for each feature.",
        "nunique": "The unique values plot displays the number of unique values for each feature in the dataset on a logarithmic scale. This helps in understanding the cardinality of each feature and identifying features with high or low variability."
    }

    # Improved Correlation matrix with better readability
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f', annot_kws={"size": 8})
    plt.title('Correlation Matrix', fontsize=16)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.figtext(0.5, -0.05, descriptions["correlation_matrix"], wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(os.path.join(file_path, "correlation_matrix.png"), bbox_inches='tight')
    plt.clf()

    # Skewness with improved visualization and logarithmic scale
    skewness = df.skew().sort_values()
    plt.figure(figsize=(14, 6))
    sns.barplot(x=skewness.index, y=skewness.values, palette='viridis', hue=skewness.index, dodge=False, legend=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.yscale('log')
    plt.ylabel('Log Skewness', fontsize=12)
    plt.title('Skewness of Features', fontsize=16)
    plt.tight_layout()
    plt.figtext(0.5, -0.1, descriptions["skewness"], wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(os.path.join(file_path, "skewness.png"), bbox_inches='tight')
    plt.clf()

    # Kurtosis with improved visualization and logarithmic scale
    kurtosis = df.kurt().sort_values()
    plt.figure(figsize=(14, 6))
    sns.barplot(x=kurtosis.index, y=kurtosis.values, palette='magma', hue=kurtosis.index, dodge=False, legend=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.yscale('log')
    plt.ylabel('Log Kurtosis', fontsize=12)
    plt.title('Kurtosis of Features', fontsize=16)
    plt.tight_layout()
    plt.figtext(0.5, -0.1, descriptions["kurtosis"], wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(os.path.join(file_path, "kurtosis.png"), bbox_inches='tight')
    plt.clf()

    # Descriptive statistics table with improved readability and shortened timestamp values
    desc = df.describe().transpose()
    if 'timestamp' in desc.index:
        desc.loc['timestamp'] = desc.loc['timestamp'].apply(shorten_large_numbers)
    desc = desc.applymap(str)
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=desc.values, colLabels=desc.columns, rowLabels=desc.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.5, 1.5)
    plt.title('Descriptive Statistics', fontsize=16)
    plt.tight_layout()
    plt.figtext(0.5, -0.1, descriptions["descriptive_statistics_table"], wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(os.path.join(file_path, "descriptive_statistics_table.png"), bbox_inches='tight')
    plt.clf()

    # Unique values with logarithmic scale
    nunique = df.nunique().sort_values()
    plt.figure(figsize=(14, 6))
    sns.barplot(x=nunique.index, y=nunique.values, palette='coolwarm', hue=nunique.index, dodge=False, legend=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.yscale('log')
    plt.ylabel('Log Number of Unique Values', fontsize=12)
    plt.title('Number of Unique Values in Each Column', fontsize=16)
    plt.tight_layout()
    plt.figtext(0.5, -0.1, descriptions["nunique"], wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(os.path.join(file_path, "nunique.png"), bbox_inches='tight')
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
