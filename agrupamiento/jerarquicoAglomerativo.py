import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
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
df_muestra = df.sample(n=6000, random_state=semilla)

# ------------------- Preparación de datos -------------------

def preparar_datos(df_sample):
    df_sel = df_sample[variables].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sel)
    return X_scaled, df_sel

X_scaled, df_sel = preparar_datos(df_muestra)

# ------------------- Dendrograma -------------------

# Muestra reducida para visualización clara del dendrograma
X_sub = X_scaled[:500]  # 500 puntos es manejable
Z = linkage(X_sub, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    truncate_mode='level',  # Solo mostrar niveles superiores
    p=25,  # profundidad máxima del dendrograma
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True
)
plt.title("Dendrograma - Método de Ward")
plt.xlabel("Muestras (truncadas)")
plt.ylabel("Distancia")
plt.tight_layout()
plt.show()

# ------------------- Agglomerative Clustering -------------------

modelo = AgglomerativeClustering(n_clusters=4, linkage='ward')
etiquetas = modelo.fit_predict(X_scaled)
df_sel['Cluster'] = etiquetas

# ------------------- Visualización PCA -------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_sel['Cluster'], palette='tab10', s=10)
plt.title("Clusterización Jerárquica (PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster", loc="best", bbox_to_anchor=(1, 1))
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
ax.set_title("Clusterización Jerárquica 3D")
ax.set_xlabel("Edad del atleta")
ax.set_ylabel("Velocidad calculada (km/h)")
ax.set_zlabel("Distancia (km)")
legend_labels = [f'Cluster {i}' for i in np.unique(etiquetas)]
ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Cluster", loc='best')
plt.tight_layout()
plt.show()

# ------------------- Métricas Internas -------------------

print("\nResumen de métricas internas para agrupamiento jerárquico (k=4):\n")
sil = silhouette_score(X_scaled, etiquetas)
calinski = calinski_harabasz_score(X_scaled, etiquetas)
davies = davies_bouldin_score(X_scaled, etiquetas)

print(f"Silhouette Score: {sil:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.1f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
