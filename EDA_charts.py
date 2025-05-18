import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el archivo filtrado
#file_path = "TWO_CENTURIES_FILTERED.csv"
#file_path = "TWO_CENTURIES_FINAL_WITH_SEASON.csv"
file_path ="TWO_CENTURIES_CLEAN_FINAL.csv"
df = pd.read_csv(file_path)

# 2. Agrupar por año de evento y contar registros
event_counts = df['Year of event'].value_counts().sort_index()

# 3. Graficar
plt.figure(figsize=(12, 6))
plt.plot(event_counts.index, event_counts.values, marker='o', linestyle='-')
plt.title('Cantidad de registros por año')
plt.xlabel('Año')
plt.ylabel('Número de registros')
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_registros_por_año.png")
plt.show()

print("\n Gráfico generado: grafico_registros_por_año.png")

#________________________________________________________________________________________________
# 1. Calcular cantidad de valores faltantes por columna
missing_counts = df.isnull().sum()

# 2. Mostrar solo las columnas con valores faltantes
missing_with_values = missing_counts[missing_counts > 0]

print("\nValores faltantes por columna:\n")
print(missing_with_values.sort_values(ascending=False))


# Graficar valores faltantes
if not missing_with_values.empty:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    missing_with_values.sort_values(ascending=True).plot(kind='barh', color='tomato')
    plt.title("Cantidad de valores faltantes por columna")
    plt.xlabel("Número de valores faltantes")
    plt.tight_layout()
    plt.savefig("grafico_valores_faltantes.png")
    plt.show()
    print("\nGráfico guardado como: grafico_valores_faltantes.png")
else:
    print("\nNo hay valores faltantes en ninguna columna. No se genera gráfico.")

print("\nGráfico guardado como: grafico_valores_faltantes.png")