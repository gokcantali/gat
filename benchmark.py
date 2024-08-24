import time

from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y


TEST_RATIO = 0.10
VALIDATION_RATIO = 0.30
TRIALS = 10


class RandomForestConfig:
    max_depth = 4
    number_of_estimators = 100

    def __init__(self, **kwargs):
        if 'max_depth' in kwargs:
            self.max_depth = kwargs['max_depth']
        if 'n_estimators' in kwargs:
            self.n_estimators = kwargs['n_estimators']

def report_cm_results(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "accuracy": (tp + tn) / (tp + tn + fn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def train_random_forest(train_data, train_label, **kwargs):
    clf = RandomForestClassifier(**kwargs)
    clf.fit(train_data, train_label)
    return clf


def test_random_forest(clf, test_data, test_label):
    predicted_label = clf.predict(test_data)
    return confusion_matrix(test_label, predicted_label)


def train_svm(train_data, train_label, **kwargs):
    clf = LinearSVC(**kwargs)
    clf.fit(train_data, train_label)
    return clf


def test_svm(clf, test_data, test_label):
    predicted_label = clf.predict(test_data)
    return confusion_matrix(test_label, predicted_label)


def train_random_forest_with_k_fold_cv(X_train_val, y_train_val, is_verbose=True):
    parameters = {
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "n_estimators": [20, 50, 75, 100, 150, 250, 400],
    }
    n_splits = int((1 - TEST_RATIO) / VALIDATION_RATIO)

    start_time = time.time()
    skf = StratifiedKFold(n_splits=n_splits)

    best_score = 0.0
    best_config = {}
    for max_depth in parameters["max_depth"]:
        for n_estimators in parameters["n_estimators"]:
            rfc = RandomForestClassifier(
                max_depth=max_depth, n_estimators=n_estimators
            )

            scores = []
            for ind, (train_ind, val_ind) in enumerate(skf.split(X_train_val, y_train_val)):
                X_train = X_train_val[X_train_val.index.isin(train_ind)]
                y_train = y_train_val[y_train_val.index.isin(train_ind)]
                X_val = X_train_val[X_train_val.index.isin(val_ind)]
                y_val = y_train_val[y_train_val.index.isin(val_ind)]

                rfc.fit(X_train, y_train)

                score = rfc.score(X_val, y_val)
                scores.append(score)

            if mean(scores) > best_score:
                best_score = mean(scores)
                best_config = {
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                }

    end_time = time.time()

    if is_verbose:
        print(f"Cross validation time: {end_time - start_time} seconds")
        print("Best Parameters based on Grid Search:")
        print(best_config)

    clf = RandomForestClassifier(**best_config)
    clf.fit(X_train_val, y_train_val)

    return clf


def train_svm_with_k_fold_cv(X_train_val, y_train_val, is_verbose=True):
    parameters = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "tol": [1e-4, 1e-3],
    }
    n_splits = int((1 - TEST_RATIO) / VALIDATION_RATIO)
    skf = StratifiedKFold(n_splits=n_splits)

    start_time = time.time()

    best_score = 0.0
    best_config = {}
    for c in parameters["C"]:
        for penalty in parameters["penalty"]:
            for tol in parameters["tol"]:
                svc = LinearSVC(
                    C=c,
                    penalty=penalty,
                    tol=tol,
                    dual=True
                )

                scores = []
                for ind, (train_ind, val_ind) in enumerate(skf.split(X_train_val, y_train_val)):
                    X_train = X_train_val[X_train_val.index.isin(train_ind)]
                    y_train = y_train_val[y_train_val.index.isin(train_ind)]
                    X_val = X_train_val[X_train_val.index.isin(val_ind)]
                    y_val = y_train_val[y_train_val.index.isin(val_ind)]

                    svc.fit(X_train, y_train)

                    score = svc.score(X_val, y_val)
                    scores.append(score)

                if mean(scores) > best_score:
                    best_score = mean(scores)
                    best_config = {
                        "C": c,
                        "penalty": penalty,
                        "tol": tol
                    }

    end_time = time.time()

    if is_verbose:
        print(f"Cross validation time: {end_time - start_time} seconds")
        print("Best Parameters based on Grid Search:")
        print(best_config)

    svc = LinearSVC(**best_config, dual=True)
    svc.fit(X_train_val, y_train_val)

    return svc


if __name__ == '__main__':
    #RANDOM_STATE = 42

    df = preprocess_df(use_diversity_index=True)
    X = preprocess_X(df, use_diversity_index=True)
    X = X.drop(columns=['source_pod_label_normalized'])
    X = X.drop(columns=['destination_pod_label_normalized'])
    # X = X.drop(columns=['source_namespace_label_normalized'])
    # X = X.drop(columns=['destination_namespace_label_normalized'])
    # X = X.drop(columns=['diversity_index'])
    # X = X.drop(columns=['ack_flag'])
    # X = X.drop(columns=['psh_flag'])
    y = preprocess_y(df)

    for _ in range(TRIALS):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_RATIO,
            stratify=y, #random_state=RANDOM_STATE
        )
        # rf = train_random_forest_with_k_fold_cv(
        #     X_train_val, y_train_val, is_verbose=True
        # )

        # rf = train_random_forest(
        #     X_train_val, y_train_val,
        #     max_depth=6, n_estimators=100
        # )
        # cm = test_random_forest(rf, X_test, y_test)
        # print(report_cm_results(cm))

        # svm = train_svm(X_train, y_train, dual='auto')
        # cm = test_svm(svm, X_test, y_test)
        # print(report_cm_results(cm))

        svc = train_svm_with_k_fold_cv(
            X_train_val, y_train_val, is_verbose=True
        )
        cm = test_svm(svc, X_test, y_test)
        print(report_cm_results(cm))
