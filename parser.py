import copy
import re, json


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


def print_avg_performance_metrics_into_file(avg_perf_metrics: dict, f_name: str):
    class_metrics = ["pre", "rec", "f1"]
    general_metrics = [
        "acc\t\t\t", "weight_avg_pre", "simple_avg_pre",
        "weight_avg_rec", "simple_avg_rec", "weight_avg_f1", "simple_avg_f1"
    ]
    with open(f_name, "w") as file:
        file.write("Metric\t\t\t\tBenign\t\t\t\t\tDoS\t\t\t\t\t\tPort\t\t\t\t\tZap\t\t\t\t\t\tGeneral\n")
        for metric in class_metrics:
            file.write(metric+"\t\t\t\t\t")
            file.write(str(avg_perf_metrics["class"]["benign"][metric])+"\t\t")
            file.write(str(avg_perf_metrics["class"]["dos"][metric]) + "\t\t")
            file.write(str(avg_perf_metrics["class"]["port"][metric]) + "\t\t")
            file.write(str(avg_perf_metrics["class"]["zap"][metric]) + "\t\t")
            file.write("\n")
        file.write("\n")
        for metric in general_metrics:
            file.write(metric+"\t\t")
            file.write("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t")
            file.write(str(avg_perf_metrics["general"][metric.strip()]))
            file.write("\n")


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
    # average_performance_metrics = get_average_metrics_across_experiments(list_of_perf_metrics)
    # print(average_performance_metrics)
    #
    # output_file_name = "performance-results-train-worker0-test-worker0-FL-60-CF-ExpSmooth.txt"
    # print_avg_performance_metrics_into_file(
    #     average_performance_metrics, output_file_name
    # )
    suffixes = [
        "", "-FL-60-CF", "-FL-60-CF-ExpSmooth",
        "-FL-60-CF-LinRegress", "-FL-60-NON-CF"
    ]

    for train_ind in range(0, 5):
        for test_ind in range(0, 5):
            for suffix in suffixes:
                input_file_name = (
                    "results_phase2/experiment/" +
                    f"train-worker{train_ind}-" +
                    f"test-worker{test_ind}" +
                    f"{suffix}.txt"
                )
                output_file_name = (
                    "results_phase2/summary/" +
                    f"summary-train-worker{train_ind}-" +
                    f"test-worker{test_ind}" +
                    f"{suffix}.txt"
                )

                conf_matrices = parse_file_with_confusion_matrices(input_file_name)
                perf_metrics = [
                    parse_single_confusion_matrix_and_obtain_metrics(c_matrix)
                    for c_matrix in conf_matrices
                ]
                avg_perf_metrics = get_average_metrics_across_experiments(perf_metrics)
                print_avg_performance_metrics_into_file(avg_perf_metrics, output_file_name)
