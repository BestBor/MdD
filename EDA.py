import pandas as pd
import numpy as np
import re
from datetime import datetime

# --- 1. Cargar CSV con manejo de tipo mixto ---
file_path = "TWO_CENTURIES_OF_UM_RACES.csv"
df = pd.read_csv(file_path, low_memory=False)

# --- 2. Extraer número de etapas ---
def extract_stages(dist):
    if pd.isna(dist):
        return 1
    dist = str(dist).lower()
    match = re.search(r'(\d+)\s*etappen', dist)
    if match:
        return int(match.group(1))
    return 1

df['Stages'] = df['Event distance/length'].apply(extract_stages)

unrecognized_distances = set()

# 3--- Extraer etapas desde cualquier parte de la cadena ---
def extract_stages(dist):
    if pd.isna(dist):
        return 1
    dist = str(dist).lower()
    match = re.search(r'(\d+)\s*etappen', dist)
    if match:
        return int(match.group(1))
    return 1

df['Stages'] = df['Event distance/length'].apply(extract_stages)

# --- Extraer distancia en km desde partes válidas ---
unrecognized_distances = set()

def parse_distance(dist):
    if pd.isna(dist):
        return np.nan
    dist = dist.lower().strip()

    # Si hay "/", tomamos solo la parte antes del "/"
    if '/' in dist:
        dist = dist.split('/')[0].strip()

    # Soporta "100k" o "45k"
    if dist.endswith('k') and dist[:-1].replace('.', '', 1).isdigit():
        return float(dist[:-1])

    # Soporta "km"
    if 'km' in dist:
        number = re.findall(r'[\d.]+', dist)
        try:
            return float(number[0])
        except:
            unrecognized_distances.add(dist)
            return np.nan

    # Soporta "mi" o "m" para millas
    if 'mi' in dist or dist.endswith('m'):
        number = re.findall(r'[\d.]+', dist)
        try:
            return float(number[0]) * 1.60934
        except:
            unrecognized_distances.add(dist)
            return np.nan

    # Si no coincide con nada, marcar como no reconocida
    unrecognized_distances.add(dist)
    return np.nan

# Aplicar
df['Distance_km'] = df['Event distance/length'].apply(parse_distance)

# Reporte
if unrecognized_distances:
    print("\n  Valores no reconocidos en 'Event distance/length':")
    for val in sorted(unrecognized_distances):
        print(f"- {val}")
else:
    print("Todas las distancias fueron reconocidas correctamente.")

# Eliminar registros sin distancia válida
original_len = len(df)
df = df[~df['Distance_km'].isna()]
print(f"\n🧹 Filas eliminadas por distancia inválida: {original_len - len(df)}")

# --- 4. Velocidad promedio a número ---
df['Avg_speed_kmh'] = pd.to_numeric(df['Athlete average speed'], errors='coerce')

# --- 5. Extraer país desde nombre del evento (último paréntesis) ---
def extract_country(event_name):
    if pd.isna(event_name):
        return np.nan
    matches = re.findall(r'\(([^)]+)\)', event_name)
    return matches[-1] if matches else np.nan

df['Event_country'] = df['Event name'].apply(extract_country)

# --- 6. Estación del año desde la fecha ---
def get_season(date_str):
    if pd.isna(date_str):
        return np.nan
    try:
        date_obj = datetime.strptime(date_str.strip(), "%d.%m.%Y")
        month = date_obj.month
        if month in [12, 1, 2]:
            return 'Invierno'
        elif month in [3, 4, 5]:
            return 'Primavera'
        elif month in [6, 7, 8]:
            return 'Verano'
        else:
            return 'Otoño'
    except:
        return np.nan

df['Season'] = df['Event dates'].apply(get_season)

# --- 7. Calcular edad del atleta ---
df['Athlete age'] = df['Year of event'] - df['Athlete year of birth']

# --- 8. Guardar nuevo archivo transformado ---
output_path = "TWO_CENTURIES_TRANSFORMED.csv"
df.to_csv(output_path, index=False)

# --- 9. Mostrar vista previa de columnas nuevas ---
print("\nTransformación completada. Vista previa:")
print(df[['Distance_km', 'Stages', 'Avg_speed_kmh', 'Event_country', 'Season', 'Athlete age']].head())
