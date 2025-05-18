import pandas as pd
import numpy as np
import re

# 1. Cargar archivo limpio
df = pd.read_csv("TWO_CENTURIES_CLEAN_FINAL.csv")

# 2. Función para convertir 'HH:MM:SS h' → horas decimales
def parse_time_to_hours(time_str):
    if pd.isna(time_str):
        return np.nan
    time_str = time_str.strip().lower().replace(' h', '')
    match = re.match(r'^(\d{1,2}):(\d{2}):(\d{2})$', time_str)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours + minutes / 60 + seconds / 3600
    else:
        return np.nan

# 3. Aplicar conversión
df['Time_hours'] = df['Athlete performance'].apply(parse_time_to_hours)

# 4. Calcular velocidad basada en distancia y tiempo
df['Speed_calc_kmh'] = df['Distance_km'] / df['Time_hours']

# 5. Ver diferencia con la velocidad registrada
df['Speed_diff'] = df['Speed_calc_kmh'] - df['Avg_speed_kmh']

# 6. Guardar el nuevo archivo con columnas adicionales
df.to_csv("TWO_CENTURIES_TIMES_ANALYSIS.csv", index=False)

# 7. Mostrar resumen
print("\n Archivo guardado como TWO_CENTURIES_TIMES_ANALYSIS.csv")
print("Ejemplos:")
print(df[['Distance_km', 'Athlete performance', 'Time_hours', 'Avg_speed_kmh', 'Speed_calc_kmh', 'Speed_diff']].head())
