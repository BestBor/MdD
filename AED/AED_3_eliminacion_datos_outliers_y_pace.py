import pandas as pd

# Cargar el archivo con tiempos y velocidad corregidos
df = pd.read_csv("TWO_CENTURIES_ANALYSIS_CLEAN.csv")

# Filtrar por edad entre 18 y 80 años
df_filtered = df[(df['Athlete age'] >= 18) & (df['Athlete age'] <= 80)]

# Filtrar por velocidad máxima razonable (< 25 km/h)
df_filtered = df_filtered[(df_filtered['Speed_calc_kmh'] <= 25) &(df_filtered['Speed_calc_kmh'] >= 2.5)]

# Crear columna de pace (min/km) = 60 / velocidad en km/h
df_filtered['Pace_min_per_km'] = 60 / df_filtered['Speed_calc_kmh']

# Guardar archivo limpio con pace incluido
df_filtered.to_csv("TWO_CENTURIES_ANALYSIS_CLEAN_WITH_PACE.csv", index=False)

# Resumen
print(f" Registros finales: {len(df_filtered)}")
print(" Guardado como: TWO_CENTURIES_ANALYSIS_CLEAN_WITH_PACE.csv")
print(" Ejemplo de columnas:")
print(df_filtered[['Athlete age', 'Speed_calc_kmh', 'Pace_min_per_km']].head())
