import pandas as pd
from datetime import datetime
import numpy as np

# Cargar el archivo filtrado
df = pd.read_csv("TWO_CENTURIES_FINAL_VALID_DATES.csv")

# Lista para guardar fechas con error de formato
invalid_dates = set()

# Intentar convertir fechas y registrar errores
def try_parse_date(date_str):
    if pd.isna(date_str):
        invalid_dates.add("NaN")
        return np.nan
    try:
        # Intentar parsear con formato correcto
        return datetime.strptime(date_str.strip(), "%d.%m.%Y")
    except:
        # Guardar fechas que fallan
        invalid_dates.add(date_str.strip())
        return np.nan

# Aplicar prueba
_ = df['Event dates'].apply(try_parse_date)

# Mostrar resultados
print(f"\nFechas con formato incorrecto: {len(invalid_dates)} únicas\n")
for val in sorted(invalid_dates):
    print(f"- {val}")
