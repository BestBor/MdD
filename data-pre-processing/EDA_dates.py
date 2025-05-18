import pandas as pd
import re

# Cargar archivo final con fechas ya corregidas
df = pd.read_csv("TWO_CENTURIES_DATES_CLEANED_FINAL.csv")

# Función para validar el formato final "dd.mm.yyyy"
def is_valid_date_format(date_str):
    if pd.isna(date_str):
        return False
    date_str = str(date_str).strip()
    return bool(re.match(r"^(0[1-9]|[12][0-9]|3[01])\.(0[1-9]|1[0-2])\.\d{4}$", date_str))

# Aplicar máscara
valid_dates_mask = df['Event dates'].apply(is_valid_date_format)

# Contar cuántos se eliminarán
removed = len(df) - valid_dates_mask.sum()
print(f"\n Registros eliminados por formato inválido de fecha: {removed}")

# Filtrar solo válidos
df_cleaned = df[valid_dates_mask]

# Guardar archivo final limpio
df_cleaned.to_csv("TWO_CENTURIES_FINAL_VALID_DATES.csv", index=False)
print("Archivo limpio guardado como TWO_CENTURIES_FINAL_VALID_DATES.csv")


df[~valid_dates_mask].to_csv("registros_con_fecha_invalida.csv", index=False)
