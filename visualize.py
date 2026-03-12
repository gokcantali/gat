import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import json

from matplotlib.patches import Patch

FONT_SIZE = 20
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
    "font.family": "Helvetica",
    "font.size": FONT_SIZE,
    "legend.fontsize": FONT_SIZE-3,
    "axes.titlesize": FONT_SIZE-1,
    "axes.labelsize": FONT_SIZE-1,
    "xtick.labelsize": FONT_SIZE-1,
    "ytick.labelsize": FONT_SIZE-1,
})


def plot_dp_related_chart(data, title, filename, show_ylabel=True, num_rounds=5):
    rounds = list(range(1, num_rounds + 1))

    series_style = [
        ("red",   "^", "--"),
        ("blue",  "x", "--"),
        ("orange","D", "--"),
        ("cyan",  "o", "--"),
        ("purple","s", "--"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    baseline_data = data.pop("Baseline (No DP)", None)
    if baseline_data is None:
        # Create customized baseline if there is no non-DP baseline
        y_ref = 0.995
        baseline_data = [y_ref] * num_rounds

    ax.plot(
        rounds, baseline_data,
        linestyle="-", color="green", label="Baseline (No DP)", linewidth=2,
        marker="*", markersize=6
    )

    sorted_data_items = sorted(
        data.items(),
        key=lambda item: float(item[0].split("=")[1].split("$$")[0])
    )

    min_y, max_y = 1.0, 0.0
    for (label, y), (color, marker, ls) in zip(sorted_data_items, series_style):
        if min(y) < min_y:
            min_y = min(y)
        if max(y) > max_y:
            max_y = max(y)

        ax.plot(
            rounds, y, linestyle=ls, marker=marker,
            color=color, linewidth=2, markersize=6, label=label
        )

    ax.set_title(title)
    ax.set_xlabel("FL Rounds")
    if show_ylabel:
        ax.set_ylabel("F1-Score (\%)")
    ax.grid(True, linewidth=0.6, alpha=0.5)
    ax.legend(loc="lower right")
    ax.set_xlim(1, num_rounds)
    ax.set_ylim(min_y-0.03, max_y+0.03)
    plt.tight_layout()
    plt.savefig(f"./results/differential-privacy/{filename}", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_all_dp_charts():
    DP_RESULTS = [{
        "data": {
            "$$\epsilon=0.001$$": [0.72, 0.92, 0.94, 0.955, 0.960],   # red
            "$$\epsilon=0.01$$":  [0.80, 0.92, 0.94, 0.950, 0.960],   # blue
            "$$\epsilon=0.02$$":  [0.905, 0.945, 0.955, 0.958, 0.962],# orange
            "$$\epsilon=0.10$$":  [0.895, 0.944, 0.955, 0.960, 0.965] # cyan
        },
        "title": "$$C=0.9 , S=0.8 , \delta=0.02$$",
        "filename": "DP-Budget.pdf"
    },{
        "data": {
            "$$S=1.0$$": [0.65, 0.90, 0.94, 0.95, 0.95],   # red
            "$$S=0.8$$": [0.93, 0.94, 0.945, 0.950, 0.955],# blue
            "$$S=0.5$$": [0.60, 0.90, 0.94, 0.95, 0.955],  # orange
            "$$S=0.2$$": [0.18, 0.70, 0.90, 0.93, 0.95]    # cyan
        },
        "title": "$$C=1.0 , \epsilon=0.0001 , \delta=0.0001$$",
        "filename": "DP-Sensitivity.pdf"
    },{
        "data": {
            "$$C=1.0$$": [0.16, 0.70, 0.90, 0.94, 0.95],   # red
            "$$C=0.8$$": [0.89, 0.94, 0.945, 0.950, 0.955],# blue
            "$$C=0.5$$": [0.35, 0.70, 0.83, 0.90, 0.92],   # orange
            "$$C=0.3$$": [0.12, 0.26, 0.40, 0.60, 0.74]    # cyan
        },
        "title": "$$S=0.2 , \epsilon=0.0001 , \delta=0.0001$$",
        "filename": "DP-Clipping.pdf"
    }]

    for result in DP_RESULTS:
        plot_dp_related_chart(**result)


def plot_all_dp_charts_with_dynamic_values(num_rounds=20):
    results_folder = "./results/differential-privacy/"
    experiments = {
        "eps": {
            "label": "\epsilon",
            "title": "$$C=0.9 , S=0.8 , \delta=0.02$$",
            "filename": "DP-Budget.pdf",
            "data": {}
        },
        "sen": {
            "label": "S",
            "title": "$$C=1.0 , \epsilon=0.0001 , \delta=0.0001$$",
            "filename": "DP-Sensitivity.pdf",
            "data": {}
        },
        "clp": {
            "label": "C",
            "title": "$$S=0.8 , \epsilon=0.10 , \delta=0.02$$",
            "filename": "DP-Clipping.pdf",
            "data": {}
        }
    }

    # Experiments with varying epsilon (privacy budget)
    for param in experiments.keys():
        for file_name in os.listdir(results_folder):
            if file_name.startswith(f"DP_{param}_") and file_name.endswith(".json"):
                with open(f"{results_folder}/{file_name}", "r") as file:
                    param_key = experiments[param]["label"]
                    param_value = file_name[len(f"DP_{param}_"):-len(".json")]
                    data_key = f"$${param_key}={param_value}$$"
                    experiments[param]["data"][data_key] = []

                    experiment_results = json.load(file).get("metrics", [])
                    _collect_experiment_results_from_metric_data(
                        data_key, experiment_results, experiments,
                        num_rounds, param
                    )

        # handle baseline values
        with open(f"{results_folder}/FL_without_DP.json", "r") as file:
            experiments[param]["data"]["Baseline (No DP)"] = []
            experiment_results = json.load(file).get("metrics", [])
            _collect_experiment_results_from_metric_data(
                "Baseline (No DP)", experiment_results, experiments,
                num_rounds, param
            )

    for result in experiments.values():
        del result["label"]  # not needed for plotting
        plot_dp_related_chart(**result, num_rounds=num_rounds)


def _collect_experiment_results_from_metric_data(data_key: str, experiment_results,
                                                 experiments: dict[str, dict[str, str | dict[Any, Any]]],
                                                 num_rounds: int, param: str) -> int:
    round = 0
    for i, round_results in enumerate(experiment_results):
        round += 1
        while round < round_results["step"]:
            # add missing intermediary rounds if there are gaps in the data
            if i > 0:
                round_metric_value = experiment_results[i - 1]["value"]
            else:
                round_metric_value = experiment_results[i + 1]["value"]
            experiments[param]["data"][data_key].append(round_metric_value)
            round += 1

        round_metric_value = round_results["value"]
        experiments[param]["data"][data_key].append(round_metric_value)

    while round < num_rounds:
        # add missing last rounds if the experiment ended early
        experiments[param]["data"][data_key].append(round_metric_value)
        round += 1


def plot_carbon_related_chart(series_list, label_list, title, filename):
    """
    This function plots graphs related to the results of carbon emission experiments.
    It uses matplotlib to create a plot with different colors for different epsilon values.
    """

    # Set font properties
    plt.rc('font', family='helvetica', size=FONT_SIZE)
    plt.rc('axes', titlesize=FONT_SIZE)
    plt.rc('axes', labelsize=FONT_SIZE)
    plt.rc('xtick', labelsize=FONT_SIZE)
    plt.rc('ytick', labelsize=FONT_SIZE)
    plt.rc('legend', fontsize=FONT_SIZE)

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


def plot_boxplots_for_fairness_analysis():
    data = {
        'exp_smooth': {
            'Node0': [34, 28, 30, 30, 36, 34, 40, 39, 36, 35, 32, 34, 32, 32, 30, 36, 37, 35, 32, 32],
            'Node1': [35, 37, 44, 38, 34, 31, 27, 22, 29, 33, 25, 36, 40, 39, 36, 34, 32, 30, 40, 30],
            'Node2': [27, 44, 25, 32, 31, 30, 33, 38, 35, 42, 30, 28, 27, 26, 28, 32, 32, 33, 27, 31],
            'Node3': [38, 32, 31, 36, 32, 34, 30, 27, 28, 21, 36, 38, 37, 39, 43, 35, 33, 38, 42, 36],
            'Node4': [34, 24, 35, 29, 32, 36, 35, 39, 37, 34, 37, 29, 29, 29, 28, 28, 31, 29, 24, 36]
        }, 'lin_reg': {
            'Node0': [27, 36, 32, 35, 17, 46, 44, 42, 38, 39, 35, 34, 34, 33, 38, 38, 34, 35, 37, 35],
            'Node1': [41, 26, 26, 38, 30, 18, 22, 20, 23, 30, 38, 33, 35, 44, 36, 41, 39, 32, 41, 39],
            'Node2': [26, 23, 36, 26, 35, 37, 36, 42, 37, 36, 27, 26, 24, 24, 21, 24, 26, 24, 26, 23],
            'Node3': [31, 35, 29, 40, 45, 23, 22, 24, 24, 27, 43, 44, 47, 40, 44, 47, 43, 45, 39, 42],
            'Node4': [40, 45, 42, 26, 38, 41, 41, 37, 43, 33, 22, 28, 25, 24, 26, 15, 23, 29, 22, 26]
        }, 'simple_avg': {
            'Node0': [39, 40, 36, 24, 38, 35, 40, 40, 43, 39, 41, 39, 40, 39, 40, 38, 29, 36, 35, 37],
            'Node1': [27, 45, 22, 37, 29, 29, 26, 31, 30, 29, 40, 40, 44, 45, 36, 40, 43, 39, 40, 48],
            'Node2': [41, 23, 36, 33, 41, 27, 25, 32, 35, 37, 21, 23, 13, 14, 23, 20, 23, 22, 21, 19],
            'Node3': [25, 28, 37, 33, 32, 41, 29, 25, 22, 19, 39, 41, 43, 43, 39, 36, 42, 42, 44, 25],
            'Node4': [45, 29, 34, 38, 25, 33, 45, 37, 35, 41, 24, 22, 25, 24, 27, 31, 28, 26, 25, 36]
        }
    }

    method_to_beautiful_name = {'simple_avg': 'Simple Average', 'exp_smooth': 'Exponential Smoothing',
                                'lin_reg': 'Linear Regression'}
    colors = {
        'simple_avg': 'lightblue',
        'exp_smooth': 'khaki',
        'lin_reg': 'lightcoral'
    }

    # Only plot nodes present in ALL use-cases (intersection)
    nodes_sets = [set(data[method].keys()) for method in method_to_beautiful_name.keys()]
    nodes = sorted(set.intersection(*nodes_sets))  # ['Node0', ..., 'Node4']

    # Base x positions (one group per node)
    base = np.arange(1, len(nodes) + 1, dtype=float)

    # Horizontal offsets for the three boxes per group
    group_width = 0.75  # total width occupied by the 3 boxes
    box_width = group_width / 3.0  # width of each box
    offsets = np.linspace(-group_width / 3, group_width / 3, num=3)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each use-case at its offset with consistent color
    legend_handles = []
    for i, method in enumerate(method_to_beautiful_name.keys()):
        values = [data[method][n] for n in nodes]
        positions = base + offsets[i]

        bp = ax.boxplot(
            values,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False
        )

        # Color all boxes for this use-case
        for patch in bp['boxes']:
            patch.set_facecolor(colors[method])
        # Optionally tint other elements to match (comment out if you prefer defaults)
        for med in bp['medians']:
            med.set_color('black')
            med.set_linewidth(1.5)
        for mean in bp['means']:
            mean.set_linewidth(1.2)
        for whisk in bp['whiskers']:
            whisk.set_linewidth(1.2)
        for cap in bp['caps']:
            cap.set_linewidth(1.2)

        legend_handles.append(
            Patch(facecolor=colors[method], edgecolor='black', label=method_to_beautiful_name[method]))

    # X-axis labels centered under each group
    ax.set_xticks(base, labels=nodes)
    ax.set_xticklabels(["Node0", "Node1", "Node2", "Node3", "Node4"])

    # ax.set_title('The number of rounds each node is selected across 20 trials, per carbon optimization method')
    ax.set_xlabel('Node')
    ax.set_ylabel('Selection Count')
    ax.legend(handles=legend_handles, title='CO Methods', ncol=1, frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/FL-carbon-experiments/fairness-analysis.pdf', dpi=300)


if __name__ == "__main__":
    # plot_all_dp_charts()
    # plot_all_fl_carbon_charts()
    plot_boxplots_for_fairness_analysis()
