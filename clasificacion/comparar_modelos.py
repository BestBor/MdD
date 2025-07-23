import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from xgboost import XGBClassifier

# 1. Argumentos por consola
parser = argparse.ArgumentParser(description="Comparación de modelos supervisados")
parser.add_argument('--guardar_modelos', action='store_true', help="Guarda los modelos entrenados como .pkl")
args = parser.parse_args()

# 2. Cargar datos
df = pd.read_csv('datos_cluster_kprototypev1.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 3. Definir variables
features = ['pace_min_per_km', 'athlete_age', 'athlete_gender', 'grouped_distances']
target = 'cluster'
X = df[features]
y = df[target]

# 4. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Preprocesamiento
numeric_features = ['pace_min_per_km', 'athlete_age']
categorical_features = ['athlete_gender', 'grouped_distances']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 6. Diccionario de modelos
modelos = {
    "Naive Bayes": GaussianNB(),
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# 7. Función para graficar matriz de confusión
def graficar_matriz(y_true, y_pred, nombre_modelo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'confusion_{nombre_modelo.replace(" ", "_")}.png')
    plt.show()

# 8. Entrenar, predecir, comparar
resultados = []

for nombre, modelo in modelos.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', modelo)
    ])

    print(f"\nEntrenando y evaluando: {nombre}")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Guardar modelo si se pide
    if args.guardar_modelos:
        joblib.dump(pipeline, f'modelo_{nombre.replace(" ", "_")}.pkl')

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    resultados.append({
        'Modelo': nombre,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    })

    # Matriz de confusión y clasificación
    graficar_matriz(y_test, y_pred, nombre)
    print(classification_report(y_test, y_pred))

# 9. Mostrar resumen
df_resultados = pd.DataFrame(resultados)
print("\nResumen comparativo de modelos:")
print(df_resultados)

# 10. Gráfico de barras comparativo
df_resultados.set_index('Modelo')[['Accuracy', 'F1-score']].plot(kind='bar', figsize=(8, 5))
plt.title("Comparación de Modelos - Accuracy y F1-score")
plt.ylabel("Puntaje")
plt.ylim(0, 1.1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('comparacion_modelos.png')
plt.show()
