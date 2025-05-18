import pandas as pd

# Cargar el archivo final con fechas ya uniformes
df = pd.read_csv("TWO_CENTURIES_FINAL_VALID_DATES.csv")

# Convertir la columna 'Event dates' a datetime
df['Event dates'] = pd.to_datetime(df['Event dates'], format="%d.%m.%Y", errors='coerce')

# Crear columna 'Season' a partir del mes
def assign_season(month):
    if month in [12, 1, 2]:
        return 'Invierno'
    elif month in [3, 4, 5]:
        return 'Primavera'
    elif month in [6, 7, 8]:
        return 'Verano'
    else:
        return 'Otoño'

df['Season'] = df['Event dates'].dt.month.apply(assign_season)

# Guardar el resultado
df.to_csv("TWO_CENTURIES_FINAL_WITH_SEASON.csv", index=False)
print(" Columna 'Season' generada y archivo guardado como TWO_CENTURIES_FINAL_WITH_SEASON.csv")
