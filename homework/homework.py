# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# ============================================================
# Paso 0: Carga de datos
# ============================================================
# Cargar los archivos comprimidos en .zip
def load_data(train_path: str, test_path: str):
    """Carga los archivos CSV (comprimidos en zip) del conjunto de entrenamiento y prueba."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Datasets cargados correctamente")
    return train_df, test_df


# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# ============================================================
# Paso 1-2: Limpieza de datos
# ============================================================
# def clean_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
#     """Limpia los datos según las reglas del enunciado."""
#     # Renombrar la columna objetivo
#     train_df.rename(columns={"default payment next month": "default"}, inplace=True)
#     test_df.rename(columns={"default payment next month": "default"}, inplace=True)

#     # Eliminar columna ID si existe
#     for df in [train_df, test_df]:
#         if "ID" in df.columns:
#             df.drop(columns=["ID"], inplace=True)

#     # Eliminar registros con valores no disponibles (EDUCATION o MARRIAGE == 0)
#     ##     train_df = train_df[(train_df["MARRIAGE"] != 0) & (train_df["EDUCATION"] != 0)]
#     train_df = train_df.loc[train_df["MARRIAGE"] != 0]
#     train_df = train_df.loc[train_df["EDUCATION"] != 0]
#     ##     test_df = test_df[(test_df["MARRIAGE"] != 0) & (test_df["EDUCATION"] != 0)]
#     test_df = test_df.loc[test_df["MARRIAGE"] != 0]
#     test_df = test_df.loc[test_df["EDUCATION"] != 0]

#     # Agrupar valores de EDUCATION > 4 como “others” (4)
#     train_df["EDUCATION"] = train_df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

#     test_df["EDUCATION"] = test_df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

#     test_df.dropna(inplace=True)
#     train_df.dropna(inplace=True)

#     # Dividir en X e y
#     x_train, y_train = train_df.drop(columns="default"), train_df["default"]
#     x_test, y_test = test_df.drop(columns="default"), test_df["default"]

#     print("Datos limpiados correctamente")
#     return x_train, y_train, x_test, y_test


def clean_data(train_df, test_df):

    train_df = train_df.rename(columns={"default payment next month": "default"})
    test_df = test_df.rename(columns={"default payment next month": "default"})

    if "ID" in train_df.columns:
        train_df = train_df.drop(columns=["ID"])
    if "ID" in test_df.columns:
        test_df = test_df.drop(columns=["ID"])

    # eliminar registros NO disponibles
    train_df = train_df[(train_df["MARRIAGE"] != 0) & (train_df["EDUCATION"] != 0)]
    test_df = test_df[(test_df["MARRIAGE"] != 0) & (test_df["EDUCATION"] != 0)]

    # EDUCATION > 4 → others (4)
    train_df["EDUCATION"] = train_df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    test_df["EDUCATION"] = test_df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    x_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    x_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    return x_train, y_train, x_test, y_test


# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# ============================================================
# Paso 3: Creación del pipeline
# ============================================================
def make_pipeline(x_train):

    CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
    NUMERIC = [col for col in x_train.columns if col not in CATEGORICAL]

    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("kbest", SelectKBest(f_classif)),
            ("pca", PCA()),
            ("clasi", MLPClassifier(max_iter=15000, random_state=21)),
        ]
    )

    return pipeline


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# ============================================================
# Paso 4: Optimización de hiperparámetros
# ============================================================
def make_grid_search(pipeline, x_train, y_train):

    param_grid = {
        "kbest__k": [20],
        "pca__n_components": [None],
        "clasi__hidden_layer_sizes": [(50, 30, 40, 60)],
        "clasi__alpha": [0.26],
        "clasi__learning_rate_init": [0.001],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(x_train, y_train)
    return grid_search


# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# ============================================================
# Paso 5: Guardar el modelo
# ============================================================
def save_estimator(model):
    """Guarda el modelo entrenado comprimido como .pkl.gz"""
    os.makedirs("files/models", exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)
    print("Modelo guardado correctamente")


# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# ============================================================
# Paso 6: Calcular y guardar métricas
# ============================================================


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """Calcula métricas y matrices de confusión para train y test."""
    metrics = []

    for x, y, label in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
        y_pred = model.predict(x)

        precision = precision_score(y, y_pred, average="binary")
        balanced_acc = balanced_accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred, average="binary")
        f1 = f1_score(y, y_pred, average="binary")

        metrics.append(
            {
                "type": "metrics",
                "dataset": label,
                "precision": precision,
                "balanced_accuracy": balanced_acc,
                "recall": recall,
                "f1_score": f1,
            }
        )
    for x, y, label in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)
        metrics.append(
            {
                "type": "cm_matrix",
                "dataset": label,
                "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
                "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
            }
        )

    return metrics


def save_metrics(metrics):
    """Calcula métricas y matrices de confusión para train y test."""

    metrics_path = "files/output"
    os.makedirs(metrics_path, exist_ok=True)

    with open("files/output/metrics.json", "w") as file:
        for metric in metrics:
            file.write(json.dumps(metric, ensure_ascii=False))
            file.write("\n")


# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# ============================================================
# Paso 7: Ejecución principal
# ============================================================
def main():
    train_df, test_df = load_data(
        "files/input/train_data.csv.zip", "files/input/test_data.csv.zip"
    )
    x_train, y_train, x_test, y_test = clean_data(train_df, test_df)

    pipeline = make_pipeline(x_train)
    model = make_grid_search(pipeline, x_train, y_train)
    save_estimator(model)
    metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics)


if __name__ == "__main__":
    main()
# ============================================================
# Tests internos
