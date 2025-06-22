import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Rutas y carga segura del CSV
ruta_base = os.path.dirname(__file__)
csv_path = os.path.abspath(os.path.join(ruta_base, '..', 'data', 'TWO_CENTURIES_ANALYSIS_CLEAN_WITH_PACE.csv'))

try:
    df = pd.read_csv(csv_path)
    print("Importación de datos exitosa")
except FileNotFoundError:
    print(f"No se encontró el archivo CSV en: {csv_path}")
    exit(1)
except Exception as e:
    print(f"Error inesperado al leer el CSV: {e}")
    exit(1)

# Variables a utilizar
variables = ['Distance_km', 'Athlete age', 'Time_hours', 'Speed_calc_kmh']

# Semillas para reproducibilidad
semillas = [42, 95, 123]

# Resultados de las métricas
resultados = []

# Función para preparar datos (selección + escalamiento)
def preparar_datos(df_sample):
    df_sel = df_sample[variables].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sel)
    return X_scaled, df_sel

# Función para aplicar KMeans y calcular métricas
def aplicar_kmeans(X, k=4):
    modelo = KMeans(n_clusters=k, random_state=0)
    etiquetas = modelo.fit_predict(X)
    sil = silhouette_score(X, etiquetas)
    calinski = calinski_harabasz_score(X, etiquetas)
    davies = davies_bouldin_score(X, etiquetas)
    return etiquetas, sil, calinski, davies

# Función para graficar los clusters con PCA
def graficar_clusters(X, etiquetas, titulo):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=etiquetas, palette='tab10', s=10)
    plt.title(titulo)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster", loc="best", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

for i, seed in enumerate(semillas):
    print(f"Procesando muestra {i+1} con semilla {seed}...")
    df_muestra = df.sample(n=50000, random_state=seed)
    X_scaled, df_sel = preparar_datos(df_muestra)
    etiquetas, sil, calinski, davies = aplicar_kmeans(X_scaled, k=4)
    resultados.append({
        'Muestra': i+1,
        'Silhouette': sil,
        'Calinski-Harabasz': calinski,
        'Davies-Bouldin': davies
    })
    graficar_clusters(X_scaled, etiquetas, f"KMeans - Muestra {i+1}")

# Mostrar resultados finales
print(pd.DataFrame(resultados))
