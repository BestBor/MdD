import pandas as pd

# Cargar archivo con columnas calculadas
df = pd.read_csv("TWO_CENTURIES_TIMES_ANALYSIS.csv")

# Filtrar: tiempo mayor a 0 y velocidad calculada finita y razonable
df_clean = df[
    (df['Time_hours'] > 0) &
    (df['Speed_calc_kmh'].notna()) &
    (df['Speed_calc_kmh'] < 100)  # velocidad realista
]

print(f" Registros antes: {len(df)}")
print(f" Registros después de limpieza: {len(df_clean)}")
print(f" Registros eliminados: {len(df) - len(df_clean)}")

# Guardar archivo limpio
df_clean.to_csv("TWO_CENTURIES_ANALYSIS_CLEAN.csv", index=False)
print(" Guardado como TWO_CENTURIES_ANALYSIS_CLEAN.csv")
