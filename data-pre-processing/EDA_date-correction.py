import pandas as pd
import re

# Cargar el archivo limpio anterior
df = pd.read_csv("TWO_CENTURIES_DATES_CLEANED.csv")

# Función para limpiar fechas con día "00"
def fix_day_zero(date_str):
    try:
        date_str = str(date_str).strip()
        # Detectar patrón 00.mm.aaaa donde mm ≠ 00
        match = re.match(r'00\.(\d{2})\.(\d{4})$', date_str)
        if match:
            mm, yyyy = match.groups()
            if mm != "00":
                return f"15.{mm}.{yyyy}"
        return date_str  # no se modifica si no coincide o si mm == 00
    except:
        return date_str

# Aplicar función
df['Event dates'] = df['Event dates'].apply(fix_day_zero)

# Guardar archivo actualizado
df.to_csv("TWO_CENTURIES_DATES_CLEANED_FINAL.csv", index=False)
print("Fechas con día '00' corregidas y archivo guardado como TWO_CENTURIES_DATES_CLEANED_FINAL.csv")
