#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena un clasificador XGBoost para predecir el cluster
asignado por K-Prototypes a partir de variables numéricas y categóricas.

Uso:
    python xgboost_classification.py datos_cluster_kprototypev1.csv \
        --test_size 0.2 \
        --seed 42 \
        --output_model xgb_model.pkl
"""

import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost para predecir cluster desde datos de K-Prototypes"
    )
    parser.add_argument('input_csv', help='CSV con datos y columna "cluster"')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proporción del set para test (por defecto 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para train_test_split (por defecto 42)')
    parser.add_argument('--output_model', default=None, help='Ruta para guardar el modelo entrenado (opcional)')
    args = parser.parse_args()

    # 1) Carga y normalización de columnas
    df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 2) Features y target
    features = ['pace_min_per_km', 'athlete_age', 'athlete_gender', 'grouped_distances']
    target = 'cluster'
    X = df[features]
    y = df[target]

    # 3) División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    # 4) Preprocesamiento
    numeric_features = ['pace_min_per_km', 'athlete_age']
    numeric_transformer = StandardScaler()

    categorical_features = ['athlete_gender', 'grouped_distances']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 5) Pipeline: preprocesamiento + modelo
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # use_label_encoder ya no es necesario (=False)
        ('classifier', XGBClassifier(eval_metric='mlogloss', random_state=args.seed))
    ])

    # 6) Entrenamiento
    print("Entrenando XGBoost...")
    clf.fit(X_train, y_train)

    # 7) Evaluación
    print("Evaluando en el set de prueba...")
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 8) Guardado (opcional)
    if args.output_model:
        joblib.dump(clf, args.output_model)
        print(f"Modelo guardado en {args.output_model}")

if __name__ == '__main__':
    main()
