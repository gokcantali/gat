import matplotlib.pyplot as plt
import numpy as np
import json

FONT_SIZE = 17
COLORS = [
    "blue",
    "red",
    "cyan",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# Set font properties
plt.rc('font', family='helvetica', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('xtick', labelsize=FONT_SIZE)
plt.rc('ytick', labelsize=FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)


def plot_histogram(values, title, ax=None):
    """ This function plots a histogram for the values of a list """

    # Count the occurrences of each value
    unique_values, counts = np.unique(values, return_counts=True)

    if ax is None:
        _, ax = plt.subplots()

    bars = ax.bar(unique_values, counts, color=COLORS, edgecolor='black')

    ax.set(title=title, xlabel='Classes', ylabel='Count')

    total = len(values)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height / total:.1%}%',
            ha='center', va='bottom'
        )

    # Adjust the y-axis limit so that the labels are not cut off
    ax.set_ylim(0, max(counts) * 1.1)


def plot_dataset_class_histograms():
    import pandas as pd

    CLASS_TO_LABEL_MAPPING = {
        0: "Benign",
        1: "DoS",
        2: "Port Scan",
        3: "ZAP Scan",
    }
    DATASETS = [
        {
            "name": "Benign Majority",
            "path": "data/subsample/traces-benign-majority-train.csv",
        },
        {
            "name": "DoS Majority",
            "path": "data/subsample/traces-dos-majority-train.csv",
        },
        {
            "name": "Port Majority",
            "path": "data/subsample/traces-port-majority-train.csv",
        },
        {
            "name": "ZAP Majority",
            "path": "data/subsample/traces-zap-majority-train.csv",
        },
        {
            "name": "Attack Majority",
            "path": "data/subsample/traces-attack-majority-train.csv",
        },
    ]

    fig = plt.figure(figsize=(16, 16))
    layout = [
        ['plot1', 'plot1', 'plot2', 'plot2'],
        ['plot3', 'plot3', 'plot4', 'plot4'],
        ['.', 'plot5', 'plot5', '.']
    ]

    ax_dict = fig.subplot_mosaic(layout)

    for ind, dataset in enumerate(DATASETS):
        df = pd.read_csv(dataset["path"])
        classes = df["anomaly_class"].values
        class_labels = pd.Series(classes).map(CLASS_TO_LABEL_MAPPING)
        plot_histogram(
            class_labels,
            dataset["name"],
            ax=ax_dict[f'plot{ind+1}']
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    file_path = "results/NATWORK-Demo/Dataset-Histogram.png"
    plt.savefig(file_path, dpi=300)

def plot_dp_related_chart(series_list, label_list, title, filename):
    """
    This function plots graphs related to the results of differential privacy experiments.
    It uses matplotlib to create a scatter plot with different colors for different epsilon values.
    """

    # Data for plotting
    t = np.arange(1, 6, 1)
    baseline = [0.99, 0.99, 0.99, 0.99, 0.99]

    fig, ax = plt.subplots()
    ax.plot(
        t, baseline,
        color="green", label="Baseline", linestyle ='-', linewidth=3
    )

    for index, series in enumerate(series_list):
        # Scatter plot for each series
        ax.scatter(
            t, series, color=COLORS[index], label=label_list[index], s=100
        )

    ax.legend(loc ='lower right')

    ax.set(xlabel='FL Round', ylabel='Accuracy (\%)',
           title=title)
    ax.grid(ls=':')
    file_path = f"results/differential-privacy/{filename}"
    plt.savefig(file_path, dpi=300)


def plot_all_dp_charts():
    DP_RESULTS = [
        {
            "series_list": [
                [0.7987, 0.9183, 0.9403, 0.9512, 0.9589],
                [0.9025, 0.9439, 0.9528, 0.9582, 0.9634],
                [0.7394, 0.8907, 0.9343, 0.9526, 0.9617],
                [0.8963, 0.9436, 0.9510, 0.9562, 0.9650]
            ],
            "label_list": [
                r"$\epsilon = 0.01$",
                r"$\epsilon = 0.02$",
                r"$\epsilon = 0.03$",
                r"$\epsilon = 0.10$"
            ],
            "title": "",
            "filename": "FL-GATAKU-epsilon.pdf"
        },
        {
            "series_list": [
                [0.1756, 0.7083, 0.8997, 0.9382, 0.9591],
                [0.6078, 0.8958, 0.9435, 0.9549, 0.9621],
                [0.9379, 0.9484, 0.9510, 0.9551, 0.9614],
                [0.7913, 0.9285, 0.9482, 0.9553, 0.9635]
            ],
            "label_list": [
                r"$S = 0.2$",
                r"$S = 0.5$",
                r"$S = 0.8$",
                r"$S = 1.0$"
            ],
            "title": "",
            "filename": "FL-GATAKU-sensitivity.pdf"
        },
        {
            "series_list": [
                [0.1408, 0.2651, 0.4335, 0.6028, 0.7302],
                [0.3831, 0.6521, 0.8211, 0.8892, 0.9139],
                [0.8895, 0.9462, 0.9475, 0.9485, 0.9505],
                [0.1756, 0.7083, 0.8997, 0.9382, 0.9591]
            ],
            "label_list": [
                r"$C = 0.3$",
                r"$C = 0.5$",
                r"$C = 0.8$",
                r"$C = 1.0$"
            ],
            "title": "",
            "filename": "FL-GATAKU-clipping.pdf"
        }
    ]

    for result in DP_RESULTS:
        plot_dp_related_chart(**result)


def plot_carbon_related_chart(series_list, label_list, title, filename):
    """
    This function plots graphs related to the results of carbon emission experiments.
    It uses matplotlib to create a plot with different colors for different epsilon values.
    """

    # Data for plotting
    t = np.arange(1, 61, 1)

    fig, ax = plt.subplots()

    for index, series in enumerate(series_list):
        # Scatter plot for each series
        ax.plot(
            t, series,
            color=COLORS[index], label=label_list[index], linewidth=3
        )

    ax.legend(loc ='lower right')

    ax.set(xlabel='FL Round', ylabel='F1-Score (\%)',
           title=title)
    ax.grid(ls=':')
    file_path = f"results/FL-carbon-experiments/{filename}"
    plt.savefig(file_path, dpi=300)


def plot_all_fl_carbon_charts():
    METHOD_TO_LABEL_MAPPING = {
        "non_cf": "No Optimization",
        "simple_avg": "Simple Averaging",
        "exp_smooth": "Exponential Smoothing",
        "lin_reg": "Linear Regression",
    }
    TARGET_METRIC = "f1_score"

    results = json.loads(open(
        'results/FL-carbon-experiments/FL-60-rounds-carbon-experiment-results.json', 'r'
    ).read())

    series_list = []
    label_list = []
    for method in results["metrics"]:
        round_scores = [0] * 60
        for round_txt in results["metrics"][method][TARGET_METRIC]:
            round_ind = int(round_txt.split("-")[1])
            round_scores[round_ind-1] = results["metrics"][method][TARGET_METRIC][round_txt]["mean"]

        series_list.append(round_scores)
        label_list.append(METHOD_TO_LABEL_MAPPING[method])

    title = ""
    filename = "FL-60-Rounds-Carbon-Reduction-Methods.pdf"
    plot_carbon_related_chart(series_list, label_list, title, filename)


if __name__ == "__main__":
    # plot_all_dp_charts()
    # plot_all_fl_carbon_charts()
    plot_dataset_class_histograms()
