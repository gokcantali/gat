import matplotlib.pyplot as plt

from gat.data_models import Metrics


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

    ax.plot(metrics.epoch_values, metrics.train_loss, marker="o", linestyle="-",
            color="b", label="Train Loss", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.val_loss, marker="s", linestyle="--",
            color="b", label="Val Loss", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.train_accuracy, marker="o", linestyle="-",
            color="r", label="Train Accuracy", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.val_accuracy, marker="s", linestyle="--",
            color="r", label="Val Accuracy", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.train_precision, marker="o", linestyle="-",
            color="g", label="Train Precision", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.val_precision, marker="s", linestyle="--",
            color="g", label="Val Precision", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.train_recall, marker="o", linestyle="-",
            color="y", label="Train Recall", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.val_recall, marker="s", linestyle="--",
            color="y", label="Val Recall", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.train_f1, marker="o", linestyle="-",
            color="m", label="Train F1-Score", markersize=5, linewidth=2)
    ax.plot(metrics.epoch_values, metrics.val_f1, marker="s", linestyle="--",
            color="m", label="Val F1-Score", markersize=5, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Metrics", fontsize=16)
    ax.set_title("Metrics per Epoch", fontsize=18)

    ax.grid(True, linestyle="--", linewidth=0.7)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig("./results/metrics_per_epoch.png")
    plt.close()
