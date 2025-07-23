import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# 1. Cargar datos
df = pd.read_csv('datos_cluster_kprototypev1.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 2. Variables
features = ['pace_min_per_km', 'athlete_age', 'athlete_gender', 'grouped_distances']
target = 'cluster'
X = df[features]
y = df[target]

# 3. Preprocesamiento
numeric_features = ['pace_min_per_km', 'athlete_age']
categorical_features = ['athlete_gender', 'grouped_distances']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 4. Diccionario de modelos
modelos = {
    "Naive Bayes": GaussianNB(),
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# 5. Métricas a evaluar
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}

# 6. Validación cruzada
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = []

for nombre, modelo in modelos.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', modelo)
    ])
    print(f'\n--- Validando modelo: {nombre}')
    resultados_modelo = {}

    for metrica, scorer in scoring.items():
        scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=scorer, n_jobs=-1)
        resultados_modelo[metrica] = scores.mean()
        print(f'  {metrica}: {scores.mean():.4f}')

    resultados_modelo['modelo'] = nombre
    resultados.append(resultados_modelo)

# 7. Mostrar resumen
df_resultados = pd.DataFrame(resultados).set_index('modelo')
print("\n✅ Promedio de métricas por validación cruzada (5-fold):")
print(df_resultados.round(4))
