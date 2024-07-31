from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from gat.preprocesser import preprocess_df, preprocess_X, preprocess_y


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


if __name__ == '__main__':
    TEST_RATIO = 0.25
    TRIALS = 10
    #RANDOM_STATE = 42

    df = preprocess_df(use_diversity_index=False)
    X = preprocess_X(df, use_diversity_index=False)
    y = preprocess_y(df)

    for _ in range(TRIALS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_RATIO,
            stratify=y, #random_state=RANDOM_STATE
        )
        #rf = train_random_forest(X_train, y_train, max_depth=4, n_estimators=150)
        #cm = test_random_forest(rf, X_test, y_test)
        svm = train_svm(X_train, y_train, dual='auto')
        cm = test_svm(svm, X_test, y_test)

        print(cm)
