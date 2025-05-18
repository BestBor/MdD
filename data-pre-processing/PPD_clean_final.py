import pandas as pd

# 1. Cargar el archivo con fechas y season ya corregidos
df = pd.read_csv("TWO_CENTURIES_FINAL_WITH_SEASON.csv")

# 2. Columnas en las que SÍ se deben eliminar registros si están vacíos
columns_required = [
    'Avg_speed_kmh',
    'Athlete age category',
    'Athlete gender',
    'Athlete average speed'
]

# 3. Eliminar registros con valores nulos en esas columnas
original_len = len(df)
df_cleaned = df.dropna(subset=columns_required)
removed = original_len - len(df_cleaned)
print(f"\nRegistros eliminados por valores faltantes (excepto club): {removed}")

# 4. Reemplazar valores vacíos en 'Athlete club' por 'Sin club'
df_cleaned['Athlete club'] = df_cleaned['Athlete club'].fillna('Sin club')

# 5. Guardar el archivo final limpio
df_cleaned.to_csv("TWO_CENTURIES_CLEAN_FINAL.csv", index=False)
print("Archivo final limpio guardado como TWO_CENTURIES_CLEAN_FINAL.csv")
