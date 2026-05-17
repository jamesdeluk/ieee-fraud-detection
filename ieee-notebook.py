import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import sys
    import json
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import (
        train_test_split,
        StratifiedKFold,
        cross_validate,
    )
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
    from xgboost import XGBClassifier
    import shap
    import joblib

    data_dir = Path("data")
    source_dir = data_dir / "source"
    model_dir = Path("models")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Helper functions
    """)
    return


@app.function
def make_threshold_summary(probabilities, true_labels):
    # Shows the business trade-off between review volume and fraud capture at common thresholds.
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    return pd.DataFrame(
        [
            {
                "threshold": threshold,
                "alerts": (probabilities >= threshold).sum(),
                "alert_rate": round((probabilities >= threshold).mean(), 3),
                "caught_fraud": (
                    (probabilities >= threshold) & (true_labels == 1)
                ).sum(),
                "missed_fraud": (
                    (probabilities < threshold) & (true_labels == 1)
                ).sum(),
                "false_positives": (
                    (probabilities >= threshold) & (true_labels == 0)
                ).sum(),
                "precision": round(
                    precision_score(true_labels, probabilities >= threshold), 3
                ),
                "reviews_per_caught_fraud": round(
                    (probabilities >= threshold).sum()
                    / max(
                        (
                            (probabilities >= threshold) & (true_labels == 1)
                        ).sum(),
                        1,
                    ),
                    1,
                ),
                "recall": round(
                    recall_score(true_labels, probabilities >= threshold), 3
                ),
                "f1": round(
                    f1_score(true_labels, probabilities >= threshold), 3
                ),
            }
            for threshold in thresholds
        ]
    )


@app.function
def summary_to_metric_dict(summary):
    # Converts a metric summary dataframe into a compact dictionary keyed by metric name.
    return {
        row["metric"]: {
            key: json_safe_value(row[key])
            for key in summary.columns
            if key != "metric"
        }
        for row in summary.to_dict(orient="records")
    }


@app.function
def records_to_json_safe(records):
    # Converts dataframe records into plain Python values so json.dump can serialise them.
    return [
        {key: json_safe_value(value) for key, value in record.items()}
        for record in records
    ]


@app.function
def json_safe_value(value):
    # Converts pandas and NumPy scalar values into plain JSON-compatible Python values.
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@app.function
def convert_categorical_to_string(data):
    # Converts categorical columns to strings and fills missing values before encoding.
    return data.astype("string").fillna("missing")


@app.cell
def _():
    setattr(sys.modules["__main__"], "convert_categorical_to_string", convert_categorical_to_string)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Setting up + EDA
    """)
    return


@app.cell
def _():
    train_transaction = pd.read_csv(source_dir / "train_transaction.csv")
    return (train_transaction,)


@app.cell
def _(train_transaction):
    mo.ui.table(train_transaction, max_columns=None)
    return


@app.cell
def _():
    train_identity = pd.read_csv(source_dir / "train_identity.csv")
    train_identity['has_identity'] = True
    return (train_identity,)


@app.cell
def _(train_identity):
    mo.ui.table(train_identity, max_columns=None)
    return


@app.cell
def _(train_identity, train_transaction):
    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    train['has_identity'] = train['has_identity'].fillna(False)
    return (train,)


@app.cell
def _(train):
    mo.ui.table(train, max_columns=None)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Feature Analysis
    """)
    return


@app.cell(hide_code=True)
def _(train):
    _target = "isFraud"
    _columns = train.columns
    _numeric_columns = train.select_dtypes(include="number").columns
    _feature_columns = [_column for _column in _columns if _column != _target]

    _mode_frame = train.mode(dropna=True)

    analysis = pd.DataFrame({
        "column": _columns,
        "dtype": train.dtypes.astype(str).values,
        "unique_values": train.nunique(dropna=False).values,
        "missing_pct": train.isna().mean().mul(100).round(1).values,
        "sample_values": [
            train[_column]
            .dropna()
            .astype("string")
            .drop_duplicates()
            .head(10)
            .tolist()
            for _column in train.columns
        ],

        # Mode is useful for spotting dominant categories or near-constant columns.
        "mode": [
            _mode_frame[_column].iloc[0] if _column in _mode_frame and not _mode_frame[_column].dropna().empty else pd.NA
            for _column in _columns
        ],
        "mode_pct": [
            (train[_column].value_counts(dropna=True, normalize=True).iloc[0] * 100).round(1)
            if not train[_column].value_counts(dropna=True).empty
            else pd.NA
            for _column in _columns
        ],

        # Numeric ranges only make sense for genuinely numeric columns.
        "min": [
            train[_column].min() if _column in _numeric_columns else pd.NA
            for _column in _columns
        ],
        "max": [
            train[_column].max() if _column in _numeric_columns else pd.NA
            for _column in _columns
        ],
        "mean": [
            train[_column].mean().round(1) if _column in _numeric_columns else pd.NA
            for _column in _columns
        ],
        "median": [
            train[_column].median().round(1) if _column in _numeric_columns else pd.NA
            for _column in _columns
        ],

        # Missingness can itself be predictive in this dataset.
        "fraud_rate_when_present": [
            train.loc[train[_column].notna(), _target].mean().round(3)
            if _column != _target
            else pd.NA
            for _column in _columns
        ],
        "fraud_rate_when_missing": [
            train.loc[train[_column].isna(), _target].mean().round(3)
            if _column != _target and train[_column].isna().any()
            else pd.NA
            for _column in _columns
        ],

        # Numeric gap gives a quick signal check for anonymous continuous/count features.
        "non_fraud_median": [
            train.loc[train[_target] == 0, _column].median().round(3)
            if _column in _numeric_columns and _column != _target
            else pd.NA
            for _column in _columns
        ],
        "fraud_median": [
            train.loc[train[_target] == 1, _column].median().round(3)
            if _column in _numeric_columns and _column != _target
            else pd.NA
            for _column in _columns
        ],
    })

    analysis["missing_fraud_gap"] = (
        pd.to_numeric(analysis["fraud_rate_when_missing"], errors="coerce")
        - pd.to_numeric(analysis["fraud_rate_when_present"], errors="coerce")
    ).abs().round(3)

    analysis["median_fraud_gap"] = (
        pd.to_numeric(analysis["fraud_median"], errors="coerce")
        - pd.to_numeric(analysis["non_fraud_median"], errors="coerce")
    ).abs().round(3)

    # analysis.sort_values(
        # ["missing_fraud_gap", "median_fraud_gap", "unique_values"],
        # ascending=[False, False, True],
    # )
    return (analysis,)


@app.cell
def _(analysis):
    analysis
    return


@app.cell
def _(analysis):
    analysis[analysis['column'].isin(['TransactionAmt', 'card1', 'card4', 'addr1', 'dist1', 'P_emaildomain', 'C1', 'D8', 'M4', 'V313', 'id02'])]
    return


@app.cell
def _(analysis):
    analysis.columns
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Column Groups
    """)
    return


@app.cell(hide_code=True)
def define_column_groups(train):
    transactions_target = "isFraud"
    transactions_display_columns = [
        _c
        for _c in ["TransactionID", "TransactionDT"]
        if _c in train.columns
    ]
    transactions_excluded_columns = [transactions_target] + transactions_display_columns
    transactions_features = [
        _c
        for _c in train.columns
        if _c not in transactions_excluded_columns
    ]

    transactions_count_features = [
        _c
        for _c in train.columns
        if _c.startswith("C") and _c[1:].isdigit()
    ]
    transactions_timedelta_features = [
        _c
        for _c in train.columns
        if _c.startswith("D") and _c[1:].isdigit()
    ]
    transactions_match_features = [
        _c
        for _c in train.columns
        if _c.startswith("M") and _c[1:].isdigit()
    ]
    transactions_vesta_features = [
        _c
        for _c in train.columns
        if _c.startswith("V") and _c[1:].isdigit()
    ]
    identity_features = [
        _c
        for _c in train.columns
        if _c.startswith("id_")
        or _c in ["DeviceType", "DeviceInfo", "has_identity"]
    ]

    transactions_basic_features = [
        _c
        for _c in transactions_features
        if _c
        not in (
            transactions_count_features
            + transactions_timedelta_features
            + transactions_match_features
            + transactions_vesta_features
            + identity_features
        )
    ]

    transactions_basic_categorical_columns = [
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
    ]
    transactions_basic_numeric_columns = [
        _c
        for _c in transactions_basic_features
        if _c not in transactions_basic_categorical_columns
    ]
    identity_categorical_columns = [
        _c
        for _c in identity_features
        if not pd.api.types.is_numeric_dtype(train[_c])
    ]
    identity_numeric_columns = [
        _c
        for _c in identity_features
        if _c not in identity_categorical_columns
    ]

    transactions_categorical_columns = (
        transactions_basic_categorical_columns
        + transactions_match_features
        + identity_categorical_columns
    )
    transactions_numeric_columns = (
        transactions_basic_numeric_columns
        + transactions_count_features
        + transactions_timedelta_features
        + transactions_vesta_features
        + identity_numeric_columns
    )
    return (
        transactions_basic_categorical_columns,
        transactions_basic_features,
        transactions_basic_numeric_columns,
        transactions_categorical_columns,
        transactions_display_columns,
        transactions_features,
        transactions_numeric_columns,
        transactions_target,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## X + y + processing
    """)
    return


@app.cell
def split_training_data(
    train,
    transactions_basic_features,
    transactions_display_columns,
    transactions_features,
    transactions_target,
):
    X = train[transactions_features]
    y = train[transactions_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1423,
        stratify=y,
    )

    X_train_basic = X_train[transactions_basic_features]
    X_test_basic = X_test[transactions_basic_features]
    X_basic = X[transactions_basic_features]
    X_test_display = train.loc[
        X_test.index, transactions_display_columns
    ]
    return (
        X,
        X_basic,
        X_test,
        X_test_basic,
        X_test_display,
        X_train,
        X_train_basic,
        y,
        y_test,
        y_train,
    )


@app.function
def make_preprocessor(
    numeric_pipeline,
    categorical_pipeline,
    numeric_columns,
    categorical_columns,
):
    # Builds a shared preprocessing structure for the chosen numeric and categorical columns.
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


@app.cell
def _():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("to_string", FunctionTransformer(convert_categorical_to_string)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return categorical_pipeline, numeric_pipeline


@app.cell
def _(
    categorical_pipeline,
    numeric_pipeline,
    transactions_basic_categorical_columns,
    transactions_basic_numeric_columns,
    transactions_categorical_columns,
    transactions_numeric_columns,
):
    preprocessor_basic = make_preprocessor(
        numeric_pipeline,
        categorical_pipeline,
        transactions_basic_numeric_columns,
        transactions_basic_categorical_columns,
    )
    preprocessor = make_preprocessor(
        numeric_pipeline,
        categorical_pipeline,
        transactions_numeric_columns,
        transactions_categorical_columns,
    )
    return preprocessor, preprocessor_basic


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Models
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Basic (LogReg)
    """)
    return


@app.cell
def _(preprocessor_basic):
    basic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_basic),
            (
                "classifier",
                LogisticRegression(max_iter=100000, class_weight="balanced"),
            ),
        ]
    )
    return (basic_pipeline,)


@app.cell
def _():
    basic_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=1423,
    )
    return (basic_cv,)


@app.cell
def _(X_train_basic, basic_cv, basic_pipeline, y_train):
    basic_cv_results = cross_validate(
        basic_pipeline,
        X_train_basic,
        y_train,
        cv=basic_cv,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=-1,
    )
    return (basic_cv_results,)


@app.cell(hide_code=True)
def _(basic_cv_results):
    basic_cv_summary = pd.DataFrame(
        {
            "metric": [
                "roc_auc",
                "average_precision",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
            "mean": [
                basic_cv_results["test_roc_auc"].mean(),
                basic_cv_results["test_average_precision"].mean(),
                basic_cv_results["test_accuracy"].mean(),
                basic_cv_results["test_precision"].mean(),
                basic_cv_results["test_recall"].mean(),
                basic_cv_results["test_f1"].mean(),
            ],
            "std": [
                basic_cv_results["test_roc_auc"].std(),
                basic_cv_results["test_average_precision"].std(),
                basic_cv_results["test_accuracy"].std(),
                basic_cv_results["test_precision"].std(),
                basic_cv_results["test_recall"].std(),
                basic_cv_results["test_f1"].std(),
            ],
        }
    )

    basic_cv_summary.round(3)
    return (basic_cv_summary,)


@app.cell
def _(X_train_basic, basic_pipeline, y_train):
    fitted_basic_pipeline = basic_pipeline.fit(X_train_basic, y_train)
    # joblib.dump(fitted_basic_pipeline, model_dir / "basic.joblib")

    # fitted_basic_pipeline = joblib.load(model_dir / "basic.joblib")
    return (fitted_basic_pipeline,)


@app.cell
def _(X_test_basic, fitted_basic_pipeline):
    basic_pred = fitted_basic_pipeline.predict(X_test_basic)
    basic_prob = fitted_basic_pipeline.predict_proba(X_test_basic)[:, 1]
    return basic_pred, basic_prob


@app.cell(hide_code=True)
def _(basic_pred, basic_prob, y_test):
    basic_holdout_summary = pd.DataFrame(
        {
            "metric": [
                "roc_auc",
                "average_precision",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
            "value": [
                roc_auc_score(y_test, basic_prob),
                average_precision_score(y_test, basic_prob),
                accuracy_score(y_test, basic_pred),
                precision_score(y_test, basic_pred),
                recall_score(y_test, basic_pred),
                f1_score(y_test, basic_pred),
            ],
        }
    )

    basic_holdout_summary.round(3)
    return (basic_holdout_summary,)


@app.cell(hide_code=True)
def _(basic_pred, y_test):
    basic_tn, basic_fp, basic_fn, basic_tp = confusion_matrix(
        y_test,
        basic_pred,
    ).ravel()

    basic_confusion_summary = pd.DataFrame(
        {
            "metric": [
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
            ],
            "count": [
                basic_tp,
                basic_tn,
                basic_fp,
                basic_fn,
            ],
        }
    )
    basic_confusion_summary["percentage"] = (
        basic_confusion_summary["count"]
        / basic_confusion_summary["count"].sum()
        * 100
    ).round(3)

    basic_confusion_summary
    return (basic_confusion_summary,)


@app.cell(hide_code=True)
def _(basic_prob, y_test):
    basic_threshold_summary = make_threshold_summary(basic_prob, y_test)
    basic_threshold_summary
    return (basic_threshold_summary,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Full (XGBoost)
    """)
    return


@app.cell
def _(preprocessor, y_train):
    # XGBoost can use a positive-class weight so fraud examples matter despite being much rarer than non-fraud examples.
    xgboost_negative_count = (y_train == 0).sum()
    xgboost_positive_count = (y_train == 1).sum()
    xgboost_scale_pos_weight = xgboost_negative_count / xgboost_positive_count

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=5000,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=1423,
                    scale_pos_weight=xgboost_scale_pos_weight,
                ),
            ),
        ]
    )
    return (full_pipeline,)


@app.cell
def _():
    full_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=1423,
    )
    return (full_cv,)


@app.cell
def _(X_train, full_cv, full_pipeline, y_train):
    full_cv_results = cross_validate(
        full_pipeline,
        X_train,
        y_train,
        cv=full_cv,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=1,
    )
    return (full_cv_results,)


@app.cell(hide_code=True)
def _(full_cv_results):
    full_cv_summary = pd.DataFrame(
        {
            "metric": [
                "roc_auc",
                "average_precision",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
            "mean": [
                full_cv_results["test_roc_auc"].mean(),
                full_cv_results["test_average_precision"].mean(),
                full_cv_results["test_accuracy"].mean(),
                full_cv_results["test_precision"].mean(),
                full_cv_results["test_recall"].mean(),
                full_cv_results["test_f1"].mean(),
            ],
            "std": [
                full_cv_results["test_roc_auc"].std(),
                full_cv_results["test_average_precision"].std(),
                full_cv_results["test_accuracy"].std(),
                full_cv_results["test_precision"].std(),
                full_cv_results["test_recall"].std(),
                full_cv_results["test_f1"].std(),
            ],
        }
    )

    full_cv_summary.round(3)
    return (full_cv_summary,)


@app.cell
def _(X_train, full_pipeline, y_train):
    fitted_full_pipeline = full_pipeline.fit(X_train, y_train)
    # joblib.dump(fitted_full_pipeline, model_dir / "full.joblib")

    # fitted_full_pipeline = joblib.load(model_dir / "full.joblib")
    return (fitted_full_pipeline,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Bayes
    """)
    return


@app.cell
def _():
    # xgboost_search_spaces = {
    #     "classifier__max_depth": Integer(3, 10),
    #     "classifier__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
    #     "classifier__subsample": Real(0.6, 1.0),
    #     "classifier__colsample_bytree": Real(0.6, 1.0),
    #     "classifier__min_child_weight": Integer(1, 20),
    #     "classifier__gamma": Real(0.0, 10.0),
    #     "classifier__reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
    #     "classifier__reg_lambda": Real(1e-8, 100.0, prior="log-uniform"),
    # }
    return


@app.cell
def _():
    # xgboost_bayes_search = BayesSearchCV(
    #     estimator=full_pipeline,
    #     search_spaces=xgboost_search_spaces,
    #     n_iter=32,
    #     scoring="average_precision",
    #     cv=full_cv,
    #     n_jobs=1,
    #     random_state=1423,
    #     verbose=1,
    #     refit=True,
    #     return_train_score=True,
    # )
    return


@app.cell
def _():
    # xgboost_bayes_search.fit(X_train, y_train)
    return


@app.cell
def _():
    # fitted_full_pipeline = xgboost_bayes_search.best_estimator_
    # xgboost_best_params = xgboost_bayes_search.best_params_
    return


@app.cell
def _():
    # xgboost_bayes_results = (
    #     pd.DataFrame(xgboost_bayes_search.cv_results_)
    #     .sort_values("rank_test_score")
    #     [
    #         [
    #             "rank_test_score",
    #             "mean_test_score",
    #             "std_test_score",
    #             "mean_train_score",
    #             "std_train_score",
    #             "params",
    #         ]
    #     ]
    #     .reset_index(drop=True)
    # )
    # xgboost_bayes_results
    return


@app.cell
def _():
    # full_cv_summary = pd.DataFrame({
    #     "metric": ["average_precision"],
    #     "mean": [xgboost_bayes_search.best_score_],
    #     "std": [xgboost_bayes_results.loc[0, "std_test_score"]],
    # })
    # full_cv_summary
    return


@app.cell
def _():
    # joblib.dump(fitted_full_pipeline, model_dir / "full_bayes.joblib")
    # fitted_full_pipeline = joblib.load(model_dir / "full_bayes.joblib")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Predictions
    """)
    return


@app.cell
def _(X_test, fitted_full_pipeline):
    full_pred = fitted_full_pipeline.predict(X_test)
    full_prob = fitted_full_pipeline.predict_proba(X_test)[:, 1]
    return full_pred, full_prob


@app.cell(hide_code=True)
def _(full_pred, full_prob, y_test):
    full_holdout_summary = pd.DataFrame(
        {
            "metric": [
                "roc_auc",
                "average_precision",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
            "value": [
                roc_auc_score(y_test, full_prob),
                average_precision_score(y_test, full_prob),
                accuracy_score(y_test, full_pred),
                precision_score(y_test, full_pred),
                recall_score(y_test, full_pred),
                f1_score(y_test, full_pred),
            ],
        }
    )
    full_holdout_summary.round(3)
    return (full_holdout_summary,)


@app.cell(hide_code=True)
def _(full_pred, y_test):
    full_tn, full_fp, full_fn, full_tp = confusion_matrix(
        y_test,
        full_pred,
    ).ravel()

    full_confusion_summary = pd.DataFrame(
        {
            "metric": [
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
            ],
            "count": [
                full_tp,
                full_tn,
                full_fp,
                full_fn,
            ],
        }
    )
    full_confusion_summary["percentage"] = (
        full_confusion_summary["count"]
        / full_confusion_summary["count"].sum()
        * 100
    ).round(3)

    full_confusion_summary
    return (full_confusion_summary,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Thresholds
    """)
    return


@app.cell(hide_code=True)
def _(full_prob, y_test):
    full_threshold_summary = make_threshold_summary(full_prob, y_test)
    full_threshold_summary
    return (full_threshold_summary,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Feature Importances
    """)
    return


@app.cell(hide_code=True)
def _(fitted_full_pipeline):
    _xgboost_classifier = fitted_full_pipeline.named_steps["classifier"]
    _preprocessor = fitted_full_pipeline.named_steps["preprocessor"]

    _numeric_features = _preprocessor.transformers_[0][2]
    _categorical_features = _preprocessor.transformers_[1][2]

    _encoder = _preprocessor.named_transformers_["categorical"].named_steps[
        "encoder"
    ]
    _categorical_feature_names = _encoder.get_feature_names_out(
        _categorical_features
    )

    _numeric_imputer = _preprocessor.named_transformers_["numeric"].named_steps[
        "imputer"
    ]
    _missing_indicator_features = [
        f"{_numeric_features[_index]}_missing"
        for _index in _numeric_imputer.indicator_.features_
    ]

    feature_names = (
        list(_numeric_features)
        + _missing_indicator_features
        + list(_categorical_feature_names)
    )

    xgboost_feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": _xgboost_classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # xgboost_feature_importance['cumulative'] = round(xgboost_feature_importance['importance'].cumsum(), 3)
    xgboost_feature_importance["importance"] = xgboost_feature_importance[
        "importance"
    ].round(3)

    xgboost_feature_importance
    return feature_names, xgboost_feature_importance


@app.cell(hide_code=True)
def _(xgboost_feature_importance):
    _top_n = 25

    _xgboost_feature_importance_ranked = xgboost_feature_importance.sort_values(
        "importance", ascending=False
    ).copy()

    _xgboost_feature_importance_ranked["rank"] = range(
        1, len(_xgboost_feature_importance_ranked) + 1
    )

    _xgboost_feature_importance_ranked["cumulative_importance"] = (
        _xgboost_feature_importance_ranked["importance"].cumsum()
        / _xgboost_feature_importance_ranked["importance"].sum()
    )

    _xgboost_feature_importance_top = _xgboost_feature_importance_ranked.head(
        _top_n
    )

    go.Figure(
        data=[
            go.Bar(
                x=_xgboost_feature_importance_top["feature"],
                y=_xgboost_feature_importance_top["importance"],
                name="Feature importance",
            ),
            go.Scatter(
                x=_xgboost_feature_importance_top["feature"],
                y=_xgboost_feature_importance_top["cumulative_importance"],
                mode="lines+markers",
                name="Cumulative importance",
                yaxis="y2",
            ),
        ],
        layout=go.Layout(
            title=f"Top {_top_n} XGBoost feature importances",
            template="plotly_white",
            xaxis=dict(
                title="Feature",
                tickangle=-60,
            ),
            yaxis=dict(title="Importance"),
            yaxis2=dict(
                title="Cumulative share of all importance",
                overlaying="y",
                side="right",
                tickformat=".0%",
                range=[0, 1],
            ),
            # Place the legend above the chart so it does not collide with the
            # secondary y-axis title on the right-hand side.
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(t=90, r=90),
            height=600,
        ),
    )
    return


@app.cell(hide_code=True)
def _(xgboost_feature_importance):
    features_list = [
        c.split("_")[0] for c in xgboost_feature_importance["feature"]
    ]
    features_list = list(dict.fromkeys(features_list))

    feature_selected = mo.ui.dropdown(options=features_list, value="V258")
    feature_selected
    return (feature_selected,)


@app.cell(hide_code=True)
def _(feature_selected, train_transaction):
    _feature = feature_selected.value
    _ = None
    try:
        _feature_fraud_profile = (
            train_transaction.assign(
                feature_bin=pd.qcut(
                    train_transaction[_feature],
                    q=10,
                    duplicates="drop",
                )
            )
            .groupby("feature_bin", observed=True)["isFraud"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "fraud_rate"})
            .reset_index()
        )

        _base_fraud_rate = train_transaction["isFraud"].mean()

        _feature_fraud_profile.assign(
            lift=lambda _df: _df["fraud_rate"] / _base_fraud_rate
        )
        _ = _feature_fraud_profile
    except TypeError:
        print("Non-numeric column")
    _
    return


@app.cell(hide_code=True)
def _(feature_selected, train_transaction):
    _feature = feature_selected.value

    _feature_fraud_profile = (
        train_transaction.groupby(_feature, dropna=False)["isFraud"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "fraud_rate"})
        .sort_values("fraud_rate", ascending=False)
    )

    _feature_fraud_profile.round(3)
    return


@app.cell(hide_code=True)
def _(feature_selected, train_transaction):
    _feature = feature_selected.value

    _is_missing_profile = (
        train_transaction.assign(is_missing=train_transaction[_feature].isna())
        .groupby("is_missing")["isFraud"]
        .agg(["count", "mean"])
    )

    _is_missing_profile.round(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### SHAP
    """)
    return


@app.cell(hide_code=True)
def _(X_test, feature_names, fitted_full_pipeline):
    _xgboost_classifier = fitted_full_pipeline.named_steps["classifier"]
    _xgboost_preprocessor = fitted_full_pipeline.named_steps["preprocessor"]

    _X_shap_sample = X_test.sample(5000, random_state=1423)
    X_shap_transformed = _xgboost_preprocessor.transform(_X_shap_sample)

    _explainer = shap.TreeExplainer(_xgboost_classifier)
    _shap_values = _explainer(X_shap_transformed)

    shap_array = _shap_values.values

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    xgboost_shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_array).mean(axis=0),
        "mean_shap": shap_array.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    xgboost_shap_importance.round(3)
    return X_shap_transformed, shap_array


@app.cell(hide_code=True)
def _(X_shap_transformed, feature_names, shap_array):
    X_shap_display = pd.DataFrame(
        X_shap_transformed.toarray()
        if hasattr(X_shap_transformed, "toarray")
        else X_shap_transformed,
        columns=feature_names,
    )

    shap.summary_plot(
        shap_array,
        X_shap_display,
        feature_names=feature_names,
        max_display=30,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Holdout data
    """)
    return


@app.cell
def _(
    X_test,
    X_test_display,
    basic_pred,
    basic_prob,
    full_pred,
    full_prob,
    transactions_basic_features,
    transactions_display_columns,
    transactions_target,
    y_test,
):
    holdout_demo = X_test_display.join(X_test)
    holdout_demo["isFraud"] = y_test
    holdout_demo["basic_prediction"] = basic_pred
    holdout_demo["basic_fraud_probability"] = basic_prob
    holdout_demo["full_prediction"] = full_pred
    holdout_demo["full_fraud_probability"] = full_prob
    deployed_holdout_columns = [
        transactions_target,
        "basic_prediction",
        "basic_fraud_probability",
        "full_prediction",
        "full_fraud_probability",
        *transactions_display_columns,
        *transactions_basic_features,
    ]
    deployed_holdout_columns = list(dict.fromkeys(deployed_holdout_columns))
    holdout_demo = holdout_demo[deployed_holdout_columns]
    holdout_demo.to_parquet(data_dir / "holdout_demo.parquet", engine="fastparquet", index=False)
    mo.ui.table(holdout_demo, max_columns=None)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Export website artefacts
    """)
    return


@app.function
def make_feature_metadata(
    train_data, feature_columns, numeric_columns, categorical_columns
):
    # Builds the feature contract used by the website and prediction API.
    metadata = []
    numeric_column_set = set(numeric_columns)
    categorical_column_set = set(categorical_columns)

    for column in feature_columns:
        column_data = train_data[column]
        column_metadata = {
            "name": column,
            "dtype": str(column_data.dtype),
            "type": "numeric"
            if column in numeric_column_set
            else "categorical",
            "required": False,
            "missing_percentage": json_safe_value(
                round(column_data.isna().mean() * 100, 3)
            ),
            "unique_values": json_safe_value(
                column_data.nunique(dropna=False)
            ),
        }

        if column in numeric_column_set:
            column_metadata.update(
                {
                    "minimum": json_safe_value(column_data.min()),
                    "maximum": json_safe_value(column_data.max()),
                    "mean": json_safe_value(round(column_data.mean(), 3)),
                    "median": json_safe_value(round(column_data.median(), 3)),
                }
            )

        if column in categorical_column_set:
            column_metadata["sample_values"] = [
                json_safe_value(value)
                for value in column_data.value_counts(dropna=True)
                .head(25)
                .index
            ]

        metadata.append(column_metadata)

    return metadata


@app.cell
def _(
    train,
    transactions_basic_categorical_columns,
    transactions_basic_features,
    transactions_basic_numeric_columns,
):
    basic_features_metadata = {
        "model_name": "basic",
        "target": "isFraud",
        "features_count": len(transactions_basic_features),
        "numeric_features": list(transactions_basic_numeric_columns),
        "categorical_features": list(transactions_basic_categorical_columns),
        "features": make_feature_metadata(
            train,
            transactions_basic_features,
            transactions_basic_numeric_columns,
            transactions_basic_categorical_columns,
        ),
    }
    return (basic_features_metadata,)


@app.cell
def _(
    train,
    transactions_categorical_columns,
    transactions_features,
    transactions_numeric_columns,
):
    full_features_metadata = {
        "model_name": "full",
        "target": "isFraud",
        "features_count": len(transactions_features),
        "numeric_features": list(transactions_numeric_columns),
        "categorical_features": list(transactions_categorical_columns),
        "features": make_feature_metadata(
            train,
            transactions_features,
            transactions_numeric_columns,
            transactions_categorical_columns,
        ),
    }
    return (full_features_metadata,)


@app.cell
def _(
    basic_confusion_summary,
    basic_cv,
    basic_cv_summary,
    basic_holdout_summary,
    basic_threshold_summary,
    full_confusion_summary,
    full_cv,
    full_cv_summary,
    full_holdout_summary,
    full_threshold_summary,
    transactions_basic_features,
    transactions_features,
    y,
    y_test,
    y_train,
):
    model_metrics = {
        "dataset": {
            "name": "IEEE-CIS Fraud Detection",
            "target": "isFraud",
            "full_train_rows": json_safe_value(len(y)),
            "train_rows": json_safe_value(len(y_train)),
            "holdout_rows": json_safe_value(len(y_test)),
            "test_size": 0.2,
            "split_strategy": "stratified train_test_split",
            "random_state": 1423,
            "train_fraud_rate": json_safe_value(round(y_train.mean(), 6)),
            "holdout_fraud_rate": json_safe_value(round(y_test.mean(), 6)),
        },
        "models": {
            "basic": {
                "model_type": "LogisticRegression",
                "features_count": len(transactions_basic_features),
                "cv": {
                    "folds": basic_cv.get_n_splits(),
                    "metrics": summary_to_metric_dict(basic_cv_summary),
                },
                "holdout": {
                    "threshold": 0.5,
                    "metrics": summary_to_metric_dict(basic_holdout_summary),
                    "confusion_matrix": summary_to_metric_dict(
                        basic_confusion_summary
                    ),
                    "thresholds": records_to_json_safe(
                        basic_threshold_summary.to_dict(orient="records")
                    ),
                },
            },
            "full": {
                "model_type": "XGBClassifier",
                "features_count": len(transactions_features),
                "cv": {
                    "folds": full_cv.get_n_splits(),
                    "metrics": summary_to_metric_dict(full_cv_summary),
                },
                "holdout": {
                    "threshold": 0.5,
                    "metrics": summary_to_metric_dict(full_holdout_summary),
                    "confusion_matrix": summary_to_metric_dict(
                        full_confusion_summary
                    ),
                    "thresholds": records_to_json_safe(
                        full_threshold_summary.to_dict(orient="records")
                    ),
                },
            },
        },
    }
    return (model_metrics,)


@app.cell
def export_final_model_artifacts(X, X_basic, basic_pipeline, full_pipeline, y):
    # Refit deployable models on every labelled training row after holdout metrics have been captured.
    model_dir.mkdir(parents=True, exist_ok=True)
    full_train_scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    final_full_pipeline = clone(full_pipeline).set_params(
        classifier__scale_pos_weight=full_train_scale_pos_weight
    )
    fitted_basic_full_train_pipeline = clone(basic_pipeline).fit(X_basic, y)
    fitted_full_train_pipeline = final_full_pipeline.fit(X, y)

    joblib.dump(fitted_basic_full_train_pipeline, model_dir / "basic.joblib")
    joblib.dump(fitted_full_train_pipeline, model_dir / "full.joblib")

    exported_model_files = [
        model_dir / "basic.joblib",
        model_dir / "full.joblib",
    ]
    return


@app.cell
def export_metadata_artifacts(
    basic_features_metadata,
    full_features_metadata,
    model_metrics,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "basic_features.json", "w") as _file:
        json.dump(basic_features_metadata, _file, indent=2)

    with open(model_dir / "full_features.json", "w") as _file:
        json.dump(full_features_metadata, _file, indent=2)

    with open(data_dir / "model_metrics.json", "w") as _file:
        json.dump(model_metrics, _file, indent=2)

    exported_json_files = [
        model_dir / "basic_features.json",
        model_dir / "full_features.json",
        data_dir / "model_metrics.json",
    ]
    exported_json_files
    return


if __name__ == "__main__":
    app.run()
