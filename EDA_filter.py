import pandas as pd

# 1. Cargar archivo transformado
file_path = "TWO_CENTURIES_TRANSFORMED.csv"
df = pd.read_csv(file_path)

# 2. Verificar años
min_event_year = df['Year of event'].min()
max_event_year = df['Year of event'].max()

min_birth_year = df['Athlete year of birth'].min()
max_birth_year = df['Athlete year of birth'].max()

print(f"\n Año de evento - mínimo: {min_event_year}, máximo: {max_event_year}")
print(f"Año de nacimiento - mínimo: {min_birth_year}, máximo: {max_birth_year}")

# 3. Filtrar según condiciones:
# - Year of event >= 1900
# - Athlete year of birth >= 1850
original_len = len(df)
df_filtered = df[(df['Year of event'] >= 1980) & (df['Athlete year of birth'] >= 1900)]

filtered_len = len(df_filtered)
removed = original_len - filtered_len

print(f"\n Registros eliminados por filtro de año: {removed}")
print(f"Registros restantes: {filtered_len}")

# 4. Guardar nuevo archivo
df_filtered.to_csv("TWO_CENTURIES_FILTERED.csv", index=False)
print("\n Archivo guardado como: TWO_CENTURIES_FILTERED.csv")
