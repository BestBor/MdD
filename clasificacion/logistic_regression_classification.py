#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena un clasificador de Regresión Logística para predecir el cluster
asignado por K-Prototypes a partir de variables numéricas y categóricas.

Uso:
    python logistic_regression_classification.py datos_cluster_kprototypev1.csv \
        --test_size 0.2 \
        --seed 42 \
        --output_model lr_model.pkl
"""

import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main():
    parser = argparse.ArgumentParser(
        description="Regresión Logística para predecir cluster desde datos de K-Prototypes"
    )
    parser.add_argument('input_csv', help='CSV con datos y columna "cluster"')
    parser.add_argument('--test_size', type=float, default=0.2, help='Tamaño del set de prueba')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para división')
    parser.add_argument('--output_model', default=None, help='Ruta para guardar modelo entrenado')
    args = parser.parse_args()

    # Cargar datos
    df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Features y target
    features = ['pace_min_per_km', 'athlete_age', 'athlete_gender', 'grouped_distances']
    target = 'cluster'
    X = df[features]
    y = df[target]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Preprocesamiento
    numeric_features = ['pace_min_per_km', 'athlete_age']
    numeric_transformer = StandardScaler()
    categorical_features = ['athlete_gender', 'grouped_distances']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto'))
    ])

    # Entrenamiento
    print("Entrenando Regresión Logística...")
    clf.fit(X_train, y_train)

    # Evaluación
    print("Evaluando en el set de prueba...")
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Guardado (opcional)
    if args.output_model:
        joblib.dump(clf, args.output_model)
        print(f"Modelo guardado en {args.output_model}")

if __name__ == '__main__':
    main()
