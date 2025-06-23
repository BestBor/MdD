import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import os
# ------------------- Carga de datos -------------------

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

# ------------------- Configuración -------------------

variables = ['Distance_km', 'Athlete age', 'Speed_calc_kmh']
semilla = 42
df_muestra = df.sample(n=50000, random_state=semilla)

# ------------------- Preparación de datos -------------------

def preparar_datos(df_sample):
    df_sel = df_sample[variables].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sel)
    return X_scaled, df_sel

X_scaled, df_sel = preparar_datos(df_muestra)

# ------------------- DBSCAN -------------------

# Parámetros iniciales (pueden ajustarse)
eps = 1.5
min_samples = 5

modelo = DBSCAN(eps=eps, min_samples=min_samples)
etiquetas = modelo.fit_predict(X_scaled)
df_sel['Cluster'] = etiquetas

# Conteo de clusters (excluye outliers si hay -1)
n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
n_ruido = list(etiquetas).count(-1)

print(f"\nDBSCAN encontró {n_clusters} clusters y {n_ruido} puntos de ruido.")

# ------------------- Métricas -------------------

if n_clusters > 1:
    sil = silhouette_score(X_scaled, etiquetas)
    print(f"Silhouette Score: {sil:.3f}")
else:
    print("No se puede calcular Silhouette Score (menos de 2 clusters).")

# ------------------- Visualización PCA -------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_sel['Cluster'], palette='tab10', s=10, legend='full')
plt.title("DBSCAN - Proyección PCA 2D")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# ------------------- Visualización 3D -------------------

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df_sel['Athlete age'],
    df_sel['Speed_calc_kmh'],
    df_sel['Distance_km'],
    c=df_sel['Cluster'],
    cmap='tab10',
    s=10,
    alpha=0.7
)
ax.set_title("Clusterización DBSCAN 3D")
ax.set_xlabel("Edad del atleta")
ax.set_ylabel("Velocidad calculada (km/h)")
ax.set_zlabel("Distancia (km)")
legend_labels = [f'Cluster {i}' if i != -1 else 'Ruido' for i in np.unique(etiquetas)]
ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Cluster", loc='best')
plt.tight_layout()
plt.show()
