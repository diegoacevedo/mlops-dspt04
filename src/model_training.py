import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    ShuffleSplit,
    cross_val_score,
    learning_curve
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from ft_engineering import build_preprocessor, _get_feature_columns, limpiar_categoricas_con_ruido_numerico

# métricas utilizadas en validación cruzada
scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def summarize_classification(y_true, y_pred):
    """
    Calcula métricas básicas de clasificación
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def build_model(
    classifier_fn,
    data_params: dict,
    test_frac: float = 0.2,
) -> dict:
    """
    Function to train a classification model

    Args:
        classifier_fn: classification function
        preprocessor (ColumnTransformer): preprocessor pipeline object
        data_params (dict): dictionary containing 'name_of_y_col',
                            'names_of_x_cols', and 'dataset'
        test_frac (float): fraction of data for the test, default 0.2

    Returns:
        dict: dictionary with the model performance metrics on train and test

    """

    # Extract data parameters
    name_of_y_col = data_params["name_of_y_col"]
    names_of_x_cols = data_params["names_of_x_cols"]
    dataset = data_params["dataset"]

    dataset = limpiar_categoricas_con_ruido_numerico(dataset, 'tendencia_ingresos',['Creciente','Decreciente','Estable'] )

    # Separate the feature columns and the target column
    X = dataset[names_of_x_cols]
    Y = dataset[name_of_y_col]

    # Split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_frac, random_state=1234
    )

    # Detecta qué columnas numéricas/categóricas/ordinales hay en x_train
    numeric_cols, categorical_cols, ordinal_cols = _get_feature_columns(x_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, ordinal_cols)

    # Create the pipeline with preprocessing and the classification model
    classifier_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier_fn)]
    )

    # Train the classifier pipeline
    model = classifier_pipe.fit(x_train, y_train)

    # Predict the test data
    y_pred = model.predict(x_test)

    # Predict the train data
    y_pred_train = model.predict(x_train)

    # Calculate the performance metrics
    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)

    kfold = KFold(n_splits=10)
    model_pipe = Pipeline(steps=[("model", model)])

    cv_results = {}
    train_results = {}

    # Ejecutamos validación cruzada
    for metric in scoring_metrics[:-1]:  
        cv_results[metric] = cross_val_score(
            model_pipe, x_train, y_train, cv=kfold, scoring=metric
        )
        # Se evalúa sobre el Dataset de pruebas
        model_pipe.fit(x_train, y_train)
        train_results[metric] = model_pipe.score(x_train, y_train)

    # Se convierten los resultados en un df
    cv_results_df = pd.DataFrame(cv_results)

    common_params = {
        "X": x_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=123),
        "n_jobs": -1,
        "return_times": True,
    }

    scoring_metric = "recall"

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        model_pipe, **common_params, scoring=scoring_metric
    )

    # Calculate the mean and standard deviation of the scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Calculate the mean and standard deviation of the fit and score times
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)

    # Plot the learning curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
    ax.plot(train_sizes, train_mean, "o-", label="Training score")
    ax.plot(train_sizes, test_mean, "o-", color="orange", label="Cross-validation score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
    ax.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.3, color="orange"
    )

    # Configure the title and labels
    ax.set_title(f"Learning Curve for {model.steps[-1][1].__class__.__name__}")
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring_metric)
    ax.legend(loc="best")

    # Show the plot
    plt.show()

    # Print the values for analysis
    print("Training Sizes:", train_sizes)
    print("Training Scores Mean:", train_mean)
    print("Training Scores Std:", train_std)
    print("Test Scores Mean:", test_mean)
    print("Test Scores Std:", test_std)

        # Plot the scalability regarding fit time and score time
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)

    # Scalability regarding the fit time
    ax[0].plot(train_sizes, fit_times_mean, "o-")
    ax[0].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.3,
    )
    ax[0].set_ylabel("Fit time (s)")
    ax[0].set_title(f"Scalability of the {model.steps[-1][1].__class__.__name__} classifier")

    # Scalability regarding the score time
    ax[1].plot(train_sizes, score_times_mean, "o-")
    ax[1].fill_between(
        train_sizes,
        score_times_mean - score_times_std,
        score_times_mean + score_times_std,
        alpha=0.3,
    )
    ax[1].set_ylabel("Score time (s)")
    ax[1].set_xlabel("Number of training samples")

    # Show the plot
    plt.show()

    # Print the fit and score times for analysis
    print("Fit Times Mean:", fit_times_mean)
    print("Fit Times Std:", fit_times_std)
    print("Score Times Mean:", score_times_mean)
    print("Score Times Std:", score_times_std)
    joblib.dump(model, "modelo_pipeline.pkl")

    return {"train": train_summary, "test": test_summary}