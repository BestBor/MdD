import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Carga y preprocesamiento de datos
# ------------------------------------
df = pd.read_csv('datos_preprocesados.csv')

# 1.1. Agrupación de distancias: <50-100, 100-200, 200-300, 300+ km>
df['distance_grouped'] = pd.cut(
    df['Distance_km'],
    bins=[50, 100, 200, 300, df['Distance_km'].max()],
    labels=['50-100km', '100-200km', '200-300km', '300+km'],
    include_lowest=True
)
print("Conteo por distance_grouped:")
print(df['distance_grouped'].value_counts().sort_index(), "\n")

# 1.2. Discretización de ritmos por cuartiles (qcut)
# Cada cuartil tendrá aproximadamente igual número de registros.
df['pace_qcat'] = pd.qcut(
    df['Pace_min_per_km'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)
print("Conteo por pace_qcat (cuartiles):")
print(df['pace_qcat'].value_counts().sort_index(), "\n")

# 2. Preparar transacciones (one-hot encoding sobre distance_grouped y pace_qcat)
one_hot = pd.get_dummies(df[['distance_grouped', 'pace_qcat']]).clip(upper=1)
transactions = one_hot.copy()

# 3. Filtrado auxiliar para asociar distance→pace
# ------------------------------------------------
def is_distance(itemset):
    return any(str(item).startswith('distance_grouped') for item in itemset)

def is_qpace(itemset):
    return any(str(item).startswith('pace_qcat') for item in itemset)

# 4. Ejecución de Apriori y reglas con parámetros ajustables
# ---------------------------------------------------------
support_values    = [0.01, 0.005, 0.001]
confidence_values = [0.3, 0.2]
lift_thresholds   = [1.0, 1.5]

for sup in support_values:
    print(f"\n=== Soporte mínimo = {sup*100:.2f}% ===")
    freq_itemsets = apriori(
        transactions,
        min_support=sup,
        use_colnames=True
    )
    if freq_itemsets.empty:
        print("No hay ítem-sets frecuentes para este soporte.")
        continue

    for conf in confidence_values:
        print(f"  - Confianza mínima = {conf*100:.0f}%:")
        rules = association_rules(
            freq_itemsets,
            metric='confidence',
            min_threshold=conf
        )
        if rules.empty:
            print("    * No se generaron reglas con este nivel de confianza.")
            continue

        # Filtrar reglas distance_grouped -> pace_qcat
        mask = (
            rules['antecedents'].apply(is_distance) &
            rules['consequents'].apply(is_qpace)
        )
        dp_rules = rules[mask].copy()
        if dp_rules.empty:
            print("    * Sin reglas distance_grouped→pace_qcat.")
            continue

        # Mostrar top reglas por lift según umbral
        for lift_th in lift_thresholds:
            subset = dp_rules[dp_rules['lift'] >= lift_th]
            if subset.empty:
                print(f"    * Ninguna con lift ≥ {lift_th}.")
            else:
                print(f"    * Reglas con lift ≥ {lift_th}:")
                print(subset.sort_values('lift', ascending=False)
                      .loc[:, ['antecedents','consequents','support','confidence','lift']]
                      .head(5))

print("\nProceso completo.")
