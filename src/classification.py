"""
classification.py
==================
Supervised learning module: trains and compares Decision Tree and
Random Forest classifiers (with an optional ANOVA feature-selection
pipeline), and provides an Optuna-based hyperparameter search across
model families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    report: str


def train_random_forest(X_train, y_train, X_test, y_test,
                         max_depth: Optional[int] = 10, random_state: int = 0) -> ModelResult:
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return ModelResult(
        name="RandomForest",
        model=clf,
        accuracy=accuracy_score(y_test, y_pred),
        report=classification_report(y_test, y_pred),
    )


def train_decision_tree(X_train, y_train, X_test, y_test) -> ModelResult:
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return ModelResult(
        name="DecisionTree",
        model=clf,
        accuracy=accuracy_score(y_test, y_pred),
        report=classification_report(y_test, y_pred),
    )


def train_random_forest_with_feature_selection(
    X_train, y_train, X_test, y_test, k: int = 20, max_depth: int = 2, random_state: int = 0
) -> ModelResult:
    """Random Forest preceded by an ANOVA F-test feature-selection step.
    Useful on high-dimensional data (e.g. flattened image pixels) where
    most features carry little signal.
    """
    k = min(k, X_train.shape[1])
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    anova_filter = SelectKBest(f_classif, k=k)
    pipeline = make_pipeline(anova_filter, clf)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return ModelResult(
        name=f"RandomForest+SelectKBest(k={k})",
        model=pipeline,
        accuracy=accuracy_score(y_test, y_pred),
        report=classification_report(y_test, y_pred),
    )


def compare_classifiers(X_train, y_train, X_test, y_test) -> list[ModelResult]:
    """Run the standard suite of classifiers and return their results,
    sorted by test accuracy (best first).
    """
    results = [
        train_decision_tree(X_train, y_train, X_test, y_test),
        train_random_forest(X_train, y_train, X_test, y_test),
        train_random_forest_with_feature_selection(X_train, y_train, X_test, y_test),
    ]
    return sorted(results, key=lambda r: r.accuracy, reverse=True)


def tune_with_optuna(X_train, y_train, X_test, y_test, n_trials: int = 50):
    """Hyperparameter / model-family search over {SVC, RandomForest,
    DecisionTree} using Optuna. Requires the optional `optuna` package
    (``pip install optuna``); raises a clear ImportError otherwise.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "tune_with_optuna requires the optional 'optuna' package. "
            "Install it with `pip install optuna`."
        ) from e

    import sklearn.svm
    import sklearn.ensemble
    from sklearn import tree

    def objective(trial: "optuna.Trial") -> float:
        classifier_name = trial.suggest_categorical(
            "classifier", ["SVC", "RandomForest", "DecisionTree"]
        )
        if classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-3, 1e3, log=True)
            clf = sklearn.svm.SVC(C=svc_c, gamma="auto")
        elif classifier_name == "DecisionTree":
            clf = tree.DecisionTreeClassifier()
        else:
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            clf = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=100
            )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study
