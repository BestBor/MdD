import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo gráfico
sns.set(style="whitegrid")
plt.rcParams["axes.titlesize"] = 14

# Cargar archivo limpio con pace
df = pd.read_csv("TWO_CENTURIES_ANALYSIS_CLEAN_WITH_PACE.csv")

# Variables numéricas a analizar
variables = ['Athlete age', 'Speed_calc_kmh', 'Time_hours', 'Distance_km', 'Pace_min_per_km']

# Histogramas
for col in variables:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col].dropna(), bins=50, kde=True, color='skyblue')
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"histograma_{col}.png")
    plt.show()

    # Histograma de Pace_min_per_km (rango ajustado)
plt.figure(figsize=(10, 5))
sns.histplot(df['Pace_min_per_km'].dropna(), bins=50, kde=True, color='skyblue')
plt.title("Histograma de Pace_min_per_km (Zoom 3–15)")
plt.xlabel("Pace (min/km)")
plt.ylabel("Frecuencia")
plt.xlim(3, 15)
plt.tight_layout()
plt.savefig("histograma_Pace_min_per_km_recortado.png")
plt.show()

# Boxplot de Pace_min_per_km (rango ajustado)
plt.figure(figsize=(8, 1.8))
sns.boxplot(x=df['Pace_min_per_km'].dropna(), color='salmon')
plt.title("Boxplot de Pace_min_per_km (Zoom 3–15)")
plt.xlabel("Pace (min/km)")
plt.xlim(3, 15)
plt.tight_layout()
plt.savefig("boxplot_Pace_min_per_km_recortado.png")
plt.show()


# Boxplots
for col in variables:
    plt.figure(figsize=(8, 1.8))
    sns.boxplot(x=df[col].dropna(), color='salmon')
    plt.title(f"Boxplot de {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")
    plt.show()
