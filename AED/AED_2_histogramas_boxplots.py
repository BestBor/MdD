import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración general
sns.set(style="whitegrid")
plt.rcParams["axes.titlesize"] = 14

# Cargar datos limpios con tiempo y velocidad calculada
df = pd.read_csv("TWO_CENTURIES_ANALYSIS_CLEAN.csv")

# Variables a graficar
variables = ['Athlete age', 'Speed_calc_kmh', 'Time_hours', 'Distance_km']

# Crear histogramas
for col in variables:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col].dropna(), bins=50, kde=True, color='skyblue')
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"histograma_{col}.png")
    plt.show()

# Crear boxplots
for col in variables:
    plt.figure(figsize=(8, 1.8))
    sns.boxplot(x=df[col].dropna(), color='salmon')
    plt.title(f"Boxplot de {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")
    plt.show()
