import matplotlib.pyplot as plt


def get_data_insights(df, file_path):
    with open(file_path, 'w') as f:
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


def plot_metrics(epoch_values, train_loss_values, val_loss_values, train_accuracy_values, val_accuracy_values,
                 train_precision_values, train_recall_values, train_f1_values,
                 val_precision_values, val_recall_values, val_f1_values):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(epoch_values, train_loss_values, marker='o', linestyle='-', color='b', label='Train Loss')
    ax.plot(epoch_values, val_loss_values, marker='s', linestyle='--', color='b', label='Val Loss')
    ax.plot(epoch_values, train_accuracy_values, marker='o', linestyle='-', color='r', label='Train Accuracy')
    ax.plot(epoch_values, val_accuracy_values, marker='s', linestyle='--', color='r', label='Val Accuracy')
    ax.plot(epoch_values, train_precision_values, marker='o', linestyle='-', color='g', label='Train Precision')
    ax.plot(epoch_values, val_precision_values, marker='s', linestyle='--', color='g', label='Val Precision')
    ax.plot(epoch_values, train_recall_values, marker='o', linestyle='-', color='y', label='Train Recall')
    ax.plot(epoch_values, val_recall_values, marker='s', linestyle='--', color='y', label='Val Recall')
    ax.plot(epoch_values, train_f1_values, marker='o', linestyle='-', color='m', label='Train F1-Score')
    ax.plot(epoch_values, val_f1_values, marker='s', linestyle='--', color='m', label='Val F1-Score')

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Metrics', fontsize=14)
    ax.set_title('Metrics per Epoch', fontsize=16)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('./results/metrics_per_epoch.png')
    plt.close()
