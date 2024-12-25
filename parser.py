import copy
import re, json


def parse_file_with_confusion_matrices(file_name):
    file_content = open(file_name, "r").read()

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

        if tp + fp == 0:
            print(
                f"WARNING: No sample for the class {label}, taking 1.0 for the metrics"
            )
            model_perf_metrics["class"][label]["pre"] = 1.0
            model_perf_metrics["class"][label]["rec"] = 1.0
            model_perf_metrics["class"][label]["f1"] = 1.0
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


def get_average_metrics_across_experiments(list_of_model_perf_metrics: list[dict]):
    classes = ["benign", "dos", "port", "zap"]
    class_metrics = {
        "pre": 0.0,
        "rec": 0.0,
        "f1": 0.0,
    }

    average_metrics = {
        "class": {label: copy.deepcopy(class_metrics) for label in classes},
        "general": {
            "acc": 0.0,
            "weight_avg_pre": 0.0,
            "simple_avg_pre": 0.0,
            "weight_avg_rec": 0.0,
            "simple_avg_rec": 0.0,
            "weight_avg_f1": 0.0,
            "simple_avg_f1": 0.0,
        },
    }

    number_of_experiments = len(list_of_model_perf_metrics)
    for perf_metric_single_experiment in list_of_model_perf_metrics:
        # compute the average of class-based metrics
        for label in classes:
            for metric_name in class_metrics:
                average_metrics["class"][label][metric_name] += (
                    perf_metric_single_experiment["class"][label][metric_name]
                    / number_of_experiments
                )

        # compute the average of general metrics
        for metric_name in list(average_metrics["general"].keys()):
            average_metrics["general"][metric_name] += (
                perf_metric_single_experiment["general"][metric_name]
                / number_of_experiments
            )

    return average_metrics


if __name__ == "__main__":
    file_name = "train-worker0-test-worker0-FL-60-CF-ExpSmooth.txt"
    conf_matrices = parse_file_with_confusion_matrices(file_name)

    print("Metrics for each Experiment:")
    list_of_perf_metrics = []
    for c_matrix in conf_matrices:
        perf_metrics = parse_single_confusion_matrix_and_obtain_metrics(c_matrix)
        print(perf_metrics)
        list_of_perf_metrics.append(perf_metrics)

    print("\nAvg Metrics across Experiments:")
    print(get_average_metrics_across_experiments(list_of_perf_metrics))
