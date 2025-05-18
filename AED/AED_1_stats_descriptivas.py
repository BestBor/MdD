import pandas as pd
import numpy as np

# Cargar el archivo corregido con tiempos y velocidades
df = pd.read_csv("TWO_CENTURIES_ANALYSIS_CLEAN.csv")

# Variables numéricas a analizar
variables = ['Time_hours', 'Speed_calc_kmh', 'Speed_diff', 'Athlete age', 'Avg_speed_kmh', 'Distance_km']

for col in variables:
    if col not in df.columns:
        print(f"\n La columna '{col}' no existe en el DataFrame.")
        continue
    
    total_count = len(df)
    missing_count = df[col].isna().sum()
    valid_count = total_count - missing_count

    serie = df[col].dropna()

    if serie.empty:
        print(f"\n No hay datos válidos en la columna '{col}' para calcular estadísticas.")
        continue

    print(f"\nEstadísticas para: {col}")
    print(f"- Total de registros: {total_count}")
    print(f"- Registros válidos usados: {valid_count}")
    print(f"- Registros ignorados por estar vacíos (NaN): {missing_count}")
    print(f"- Media: {serie.mean():.2f}")
    print(f"- Mediana: {serie.median():.2f}")
    print(f"- Moda: {serie.mode().iloc[0]:.2f}" if not serie.mode().empty else "- Moda: N/A")
    print(f"- Desviación estándar: {serie.std():.2f}")
    print(f"- Rango: {serie.min():.2f} - {serie.max():.2f}")
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    print(f"- IQR (Q3 - Q1): {q3 - q1:.2f}")
