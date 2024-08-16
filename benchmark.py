import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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
    rfc = RandomForestClassifier()
    parameters = {
        "max_depth": [3, 5],
        "n_estimators": [100, 200],
    }

    start_time = time.time()
    n_splits = int((1 - TEST_RATIO) / VALIDATION_RATIO)
    clf = GridSearchCV(rfc, parameters,
        cv=n_splits, scoring="f1", refit=True, n_jobs=8
    )
    clf.fit(X_train_val, y_train_val)
    end_time = time.time()

    if is_verbose:
        print(f"Cross validation time: {end_time - start_time} seconds")
        print("Best Parameters based on Grid Search:")
        print(sorted(clf.cv_results_))

    return clf

if __name__ == '__main__':
    #RANDOM_STATE = 42

    df = preprocess_df(use_diversity_index=True)
    X = preprocess_X(df, use_diversity_index=True)
    X = X.drop(columns=['source_pod_label_normalized'])
    X = X.drop(columns=['destination_pod_label_normalized'])
    y = preprocess_y(df)

    for _ in range(TRIALS):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_RATIO,
            stratify=y, #random_state=RANDOM_STATE
        )
        rf = train_random_forest_with_k_fold_cv(
            X_train_val, y_train_val, is_verbose=True
        )
        cm = test_random_forest(rf, X_test, y_test)
        print(report_cm_results(cm))

        # svm = train_svm(X_train, y_train, dual='auto')
        # cm = test_svm(svm, X_test, y_test)
        # print(report_cm_results(cm))

        print(cm)
