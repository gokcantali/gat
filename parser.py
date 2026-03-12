import copy
import re
import json
from typing import Optional

import numpy as np

""" The following functions are used to parse the confusion matrix files
    which are generated for the FL vs per-node experiments """

def parse_file_with_confusion_matrices(f_name: str):
    file_content = open(f_name, "r").read()

    match_pattern = r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*"
    replace_pattern = r"\1,\2,\3,\4"

    confusion_matrix_json_content = (
        "["
        + re.sub(match_pattern, replace_pattern, file_content.strip()).replace(
            "\n", ","
        )
        + "]"
    )

    return json.loads(confusion_matrix_json_content)


def parse_single_confusion_matrix_and_obtain_metrics(conf_matrix: list[list[int]]):
    classes = ["benign", "dos", "port", "zap"]
    class_metrics = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "pre": 0.0,
        "rec": 0.0,
        "f1": 0.0,
        "samples": 0,
    }

    model_perf_metrics = {
        "class": {label: copy.deepcopy(class_metrics) for label in classes},
        "general": {
            "true": 0,
            "false": 0,
            "acc": 0.0,
            "weight_avg_pre": 0.0,
            "simple_avg_pre": 0.0,
            "weight_avg_rec": 0.0,
            "simple_avg_rec": 0.0,
            "weight_avg_f1": 0.0,
            "simple_avg_f1": 0.0,
        },
    }

    # populate tp, fp, tn values for classes
    total_true = 0
    total_false = 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i == j:
                model_perf_metrics["class"][classes[i]]["tp"] += conf_matrix[i][j]
                model_perf_metrics["class"][classes[i]]["samples"] += conf_matrix[i][j]
                total_true += conf_matrix[i][j]
            else:
                model_perf_metrics["class"][classes[i]]["fn"] += conf_matrix[i][j]
                model_perf_metrics["class"][classes[i]]["samples"] += conf_matrix[i][j]
                model_perf_metrics["class"][classes[j]]["fp"] += conf_matrix[i][j]
                total_false += conf_matrix[i][j]
    total_samples = total_true + total_false

    # populate precision, recall, f1 values for classes
    for label in classes:
        tp, fp, fn = (
            model_perf_metrics["class"][label]["tp"],
            model_perf_metrics["class"][label]["fp"],
            model_perf_metrics["class"][label]["fn"],
        )

        if tp + fn == 0:
            print(
                f"WARNING: No sample for the class {label}, taking 1.0 for the metrics"
            )
            model_perf_metrics["class"][label]["pre"] = 1.0
            model_perf_metrics["class"][label]["rec"] = 1.0
            model_perf_metrics["class"][label]["f1"] = 1.0
        elif tp == 0:
            model_perf_metrics["class"][label]["pre"] = 0.0
            model_perf_metrics["class"][label]["rec"] = 0.0
            model_perf_metrics["class"][label]["f1"] = 0.0
        else:
            model_perf_metrics["class"][label]["pre"] = pre = tp / (tp + fp)
            model_perf_metrics["class"][label]["rec"] = rec = tp / (tp + fn)
            model_perf_metrics["class"][label]["f1"] = 2 * pre * rec / (pre + rec)

    # populate general metrics
    model_perf_metrics["general"]["true"] = total_true
    model_perf_metrics["general"]["false"] = total_false
    model_perf_metrics["general"]["acc"] = total_true / total_samples
    for metric_name in ["pre", "rec", "f1"]:
        for label in classes:
            model_perf_metrics["general"][f"simple_avg_{metric_name}"] += (
                model_perf_metrics["class"][label][metric_name]
                / len(classes)
            )
            model_perf_metrics["general"][f"weight_avg_{metric_name}"] += (
                model_perf_metrics["class"][label][metric_name]
                * model_perf_metrics["class"][label]["samples"]
                / total_samples
            )

    return model_perf_metrics


def get_aggregated_metrics_across_experiments(list_of_model_perf_metrics: list[dict]):
    classes = ["benign", "dos", "port", "zap"]
    class_metrics = {
        "pre": [],
        "rec": [],
        "f1": [],
    }

    aggregated_metrics = {
        "class": {label: copy.deepcopy(class_metrics) for label in classes},
        "general": {
            "acc": [],
            "weight_avg_pre": [],
            "simple_avg_pre": [],
            "weight_avg_rec": [],
            "simple_avg_rec": [],
            "weight_avg_f1": [],
            "simple_avg_f1": [],
        },
    }

    for perf_metric_single_experiment in list_of_model_perf_metrics:
        # compute the average of class-based metrics
        for label in classes:
            for metric_name in class_metrics:
                aggregated_metrics["class"][label][metric_name].append(
                    perf_metric_single_experiment["class"][label][metric_name]
                )

        # compute the average of general metrics
        for metric_name in list(aggregated_metrics["general"].keys()):
            aggregated_metrics["general"][metric_name].append(
                perf_metric_single_experiment["general"][metric_name]
            )

    return aggregated_metrics


def print_agg_performance_metrics_into_file(agg_perf_metrics: dict, f_name: str):
    class_metrics = ["pre", "rec", "f1"]
    class_labels = ["benign", "dos", "port", "zap"]

    general_metrics = [
        "acc\t\t\t", "weight_avg_pre", "simple_avg_pre",
        "weight_avg_rec", "simple_avg_rec", "weight_avg_f1", "simple_avg_f1"
    ]
    with open(f_name, "w") as file:
        file.write("Metric\t\t\tBenign\t\t\t\tDoS\t\t\t\t\tPort\t\t\t\tZap\t\t\t\t\tGeneral\n")
        for metric in class_metrics:
            file.write(metric+"\t\t\t\t")
            for label in class_labels:
                agg_metric_values = agg_perf_metrics["class"][label][metric]
                metric_mean, metric_std = np.mean(agg_metric_values), np.std(agg_metric_values)

                file.write(str(round(metric_mean, 5)))
                file.write(f" ({round(metric_std, 5):.5f})\t")
            file.write("\n")
        file.write("\n")
        for metric in general_metrics:
            agg_metric_values = agg_perf_metrics["general"][metric.strip()]
            metric_mean, metric_std = np.mean(agg_metric_values), np.std(agg_metric_values)

            file.write(metric+"\t\t")
            file.write("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t")
            file.write(str(round(metric_mean, 5)))
            file.write(f" ({round(metric_std, 5):.5f})")
            file.write("\n")


""" The following functions are used to parse the performance metrics from files
    which are generated from the FL carbon emissions experiments """
def parse_file_from_flower_output(f_name: str, folder_name: Optional[str] = None):
    if folder_name is None:
        folder_name = "FL-carbon-experiments"

    content = open(f"./results/{folder_name}/{f_name}", "r").read()
    content = content.replace("\x1b[92mINFO \x1b[0m:", "")
    content = content.replace("      ", "")
    content = content.replace("\t", "")

    metrics = {
        "loss": [], "total_emission": [],
        "accuracy": [], "precision": [], "recall": [], "f1_score": [],
        "training_f1_score": [], "selected_clients": {}, "testing_f1_score": [],
        "testing_auroc": [], "testing_pr_auc": [], "testing_recall_at_1fpr": [],
    }
    running_time = 0.0

    lines = content.split("\n")
    current_metric = None
    selected_clients_lines = []
    for index, line in enumerate(lines):
        if line.lower().startswith("info:flwr:"):
            # skip the Flower framework info lines
            continue

        if line.lower().startswith("run finished"):
            running_time = re.search("(\d+\.\d+)s", line)[1]
        else:
            for metric in metrics:
                if f"'{metric}'" in line.lower():
                    current_metric = metric
                    break
                elif "loss" in line.lower() and current_metric is None:
                    current_metric = "loss"
                    break

        if (match := re.search("round \d+: (\d+\.\d+(e-\d+)?)", line)) is not None:
            value = match[1]
            metrics[current_metric].append(value)
        elif (match := re.search("\(\d+, (\d+\.\d+(e-\d+)?)", line)) is not None:
            value = match[1]
            metrics[current_metric].append(value)

        if 'Here are the selected clients' in line:
            selected_clients_lines += [index+1, index+2, index+3]

    for line_index in selected_clients_lines:
        line = lines[line_index]
        if (match := re.search("(\d{10,25})", line)) is not None:
            client_id = match[1]
            if client_id not in metrics["selected_clients"]:
                metrics["selected_clients"][client_id] = 0
            metrics["selected_clients"][client_id] += 1

    return {
        "metrics": metrics,
        "running_time": running_time,
    }


def parse_and_aggregate_all_carbon_emission_results(
    rounds: int = 60, trials: int = 20,
    methods: list[str] = ["non_cf", "simple_avg", "exp_smooth", "lin_reg"],
    trial_prefix="",
    folder_name=None
):
    all_parsed_metrics = []
    metrics_by_method = {}
    running_time_by_method = {}

    for method in methods:
        for trial in range(1, trials+1):
            trial_text = trial if trial > 9 else f"0{trial}"
            file_name = f"FL_{rounds}_Rounds_{method}_Algorithm_{trial_prefix}{trial_text}_Trial_Results.txt"
            parsed_metrics = parse_file_from_flower_output(
                file_name,
                folder_name=folder_name
            )
            all_parsed_metrics.append(parsed_metrics["metrics"])

            if method not in metrics_by_method:
                metrics_by_method[method] = {
                    "loss": {}, "total_emission": {}, "training_f1_score": {},
                    "accuracy": {}, "precision": {}, "testing_f1_score": {}, "recall": {}, "f1_score": {},
                    "testing_auroc": {}, "testing_pr_auc": {}, "testing_recall_at_1fpr": {},
                    "selected_clients": []
                }

            for metric in metrics_by_method[method]:
                if metric == 'selected_clients':
                    metrics_by_method[method][metric].append(parsed_metrics["metrics"]["selected_clients"])
                    continue

                if metric not in parsed_metrics["metrics"]:
                    continue

                for ind, value in enumerate(parsed_metrics["metrics"][metric]):
                    round = ind + 1
                    round_key = f"round-{round}"
                    if round_key not in metrics_by_method[method][metric]:
                        metrics_by_method[method][metric][round_key] = {"values": [], "mean": 0.0, "std": 0.0}
                    metrics_by_method[method][metric][round_key]["values"].append(float(value))

            if method not in running_time_by_method:
                running_time_by_method[method] = {"values": [], "mean": 0.0, "std": 0.0}
            running_time_by_method[method]["values"].append(float(parsed_metrics["running_time"]))

        # calculates the summary statistics for each method and round
        for metric in metrics_by_method[method]:
            if metric != "selected_clients":
                for round_key in metrics_by_method[method][metric]:
                    values = metrics_by_method[method][metric][round_key]["values"]
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    metrics_by_method[method][metric][round_key]["mean"] = mean_value
                    metrics_by_method[method][metric][round_key]["std"] = std_value

        running_time_by_method[method]["mean"] = np.mean(running_time_by_method[method]["values"])
        running_time_by_method[method]["std"] = np.std(running_time_by_method[method]["values"])

    return {
        "metrics": metrics_by_method,
        "running_time": running_time_by_method,
        "all_parsed_metrics": all_parsed_metrics,
    }


### RNN-related parser functions ###
def parse_rnn_results_from_file(file_name: str):
    file_path = f"./results/VNF/{file_name}"
    with open(file_path, "r") as file:
        content = file.read()

    rnn_results = []
    for fold in content.split("Outer Classification Report\n")[1:]:
        result_str = fold.split("\n")[0].strip().replace("'", "\"")
        rnn_results.append(json.loads(result_str))

    return rnn_results


def aggregate_rnn_results(rnn_results: list[dict]):
    aggregated_results = {}

    for result in rnn_results:
        for metric in result:
            if metric == "accuracy":
                if metric not in aggregated_results:
                    aggregated_results[metric] = []
                aggregated_results[metric].append(result[metric])
            else:
                if metric not in aggregated_results:
                    aggregated_results[metric] = {}
                for sub_metric in result[metric]:
                    if sub_metric not in aggregated_results[metric]:
                        aggregated_results[metric][sub_metric] = []
                    aggregated_results[metric][sub_metric].append(result[metric][sub_metric])

    return aggregated_results


def summarize_aggregated_rnn_results(aggregated_results: dict):
    summary = {}

    for metric, values in aggregated_results.items():
        if isinstance(values, list):
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }

        else:
            summary[metric] = {}
            for sub_metric, sub_values in values.items():
                summary[metric][sub_metric] = {
                    "mean": np.mean(sub_values),
                    "std": np.std(sub_values)
                }

    return summary


# VNF-related parser functions
def parse_and_aggregate_lstm_vnf_results(rounds: int = 100, trials: int = 20):
    METHODS = ["non_cf", "simple_avg", "exp_smooth", "lin_reg"]
    metrics_by_method = {}
    running_time_by_method = {}

    for method in METHODS:
        for trial in range(1, trials+1):
            trial_text = trial if trial > 9 else f"0{trial}"
            file_name = f"FL_{rounds}_Rounds_{method}_Algorithm_LSTM_VNF_v2_{trial_text}_Trial_Results.txt"
            parsed_metrics = parse_file_from_flower_output(file_name, folder_name="VNF")

            if method not in metrics_by_method:
                metrics_by_method[method] = {
                    "validation_loss": {},
                    "total_emission": {},
                    "testing_f1_score": {},
                    "testing_auroc": {},
                    "testing_pr_auc": {},
                    "testing_recall_at_1fpr": {},
                }

            for metric in metrics_by_method[method]:
                if metric not in parsed_metrics["metrics"]:
                    continue

                for ind, value in enumerate(parsed_metrics["metrics"][metric]):
                    round = ind + 1
                    round_key = f"round-{round}"
                    if round_key not in metrics_by_method[method][metric]:
                        metrics_by_method[method][metric][round_key] = {"values": [], "mean": 0.0, "std": 0.0}
                    metrics_by_method[method][metric][round_key]["values"].append(float(value))

            if method not in running_time_by_method:
                running_time_by_method[method] = {"values": [], "mean": 0.0, "std": 0.0}
            running_time_by_method[method]["values"].append(float(parsed_metrics["running_time"]))

        # calculates the summary statistics for each method and round
        for metric in metrics_by_method[method]:
            if metric != "selected_clients":
                for round_key in metrics_by_method[method][metric]:
                    values = metrics_by_method[method][metric][round_key]["values"]
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    metrics_by_method[method][metric][round_key]["mean"] = mean_value
                    metrics_by_method[method][metric][round_key]["std"] = std_value

        running_time_by_method[method]["mean"] = np.mean(running_time_by_method[method]["values"])
        running_time_by_method[method]["std"] = np.std(running_time_by_method[method]["values"])

    return {
        "metrics": metrics_by_method,
        "running_time": running_time_by_method,
    }

if __name__ == "__main__":
    # input_file_name = "experiment/train-worker0-test-worker0-FL-60-CF-ExpSmooth.txt"
    # conf_matrices = parse_file_with_confusion_matrices(input_file_name)
    #
    # print("Metrics for each Experiment:")
    # list_of_perf_metrics = []
    # for c_matrix in conf_matrices:
    #     perf_metrics = parse_single_confusion_matrix_and_obtain_metrics(c_matrix)
    #     print(perf_metrics)
    #     list_of_perf_metrics.append(perf_metrics)
    #
    # print("\nAvg Metrics across Experiments:")
    # average_performance_metrics = get_aggregated_metrics_across_experiments(list_of_perf_metrics)
    # print(average_performance_metrics)
    #
    # output_file_name = "performance-results-train-worker0-test-worker0-FL-60-CF-ExpSmooth.txt"
    # print_agg_performance_metrics_into_file(
    #     average_performance_metrics, output_file_name
    # )

    # suffixes = [
    #     "", "-FL-60-CF", "-FL-60-CF-ExpSmooth",
    #     "-FL-60-CF-LinRegress", "-FL-60-NON-CF"
    # ]
    #
    # for train_ind in range(0, 5):
    #     for test_ind in range(0, 5):
    #         for suffix in suffixes:
    #             input_file_name = (
    #                 "results_phase2/experiment/" +
    #                 f"train-worker{train_ind}-" +
    #                 f"test-worker{test_ind}" +
    #                 f"{suffix}.txt"
    #             )
    #             output_file_name = (
    #                 "results_phase2/summary/" +
    #                 f"summary-train-worker{train_ind}-" +
    #                 f"test-worker{test_ind}" +
    #                 f"{suffix}.txt"
    #             )
    #
    #             conf_matrices = parse_file_with_confusion_matrices(input_file_name)
    #             perf_metrics = [
    #                 parse_single_confusion_matrix_and_obtain_metrics(c_matrix)
    #                 for c_matrix in conf_matrices
    #             ]
    #             aggregated_perf_metrics = get_aggregated_metrics_across_experiments(perf_metrics)
    #             print_agg_performance_metrics_into_file(aggregated_perf_metrics, output_file_name)

    results = parse_and_aggregate_all_carbon_emission_results(
        rounds=50,
        trials=20
    )
    print(results)
