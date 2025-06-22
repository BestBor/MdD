import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# 1. Carga de datos
df = pd.read_csv('datos_preprocesados.csv')

# 2. Discretización de edad por cuartiles (Q1–Q4)
# Usamos los mismos quartiles que definimos en pace para edad
# Q1: menores de 25 pct, Q2: 25–50 pct, etc.
df['age_qcut'] = pd.qcut(
    df['Athlete age'],
    q=4,
    labels=['Q1_<25pct','Q2_25-50pct','Q3_50-75pct','Q4_75-100pct']
)
print("Conteo por age_qcut (cuartiles de edad):")
print(df['age_qcut'].value_counts().sort_index(), "\n")

# 3. Discretización de distancia: 50–100, 100–200, 200–300, 300+ km
df['distance_grouped'] = pd.cut(
    df['Distance_km'],
    bins=[50, 100, 200, 300, df['Distance_km'].max()],
    labels=['50-100km','100-200km','200-300km','300+km'],
    include_lowest=True
)
print("Conteo por distance_grouped:")
print(df['distance_grouped'].value_counts().sort_index(), "\n")

# 4. One-hot encoding para age_qcut y distance_grouped
one_hot = pd.get_dummies(
    df[['age_qcut','distance_grouped']],
    prefix_sep='_'
).clip(upper=1)
transactions = one_hot.copy()

# 5. Funciones auxiliares para filtrar reglas edad→distancia
def is_age(itemset):
    return any(str(i).startswith('age_qcut') for i in itemset)

def is_distance(itemset):
    return any(str(i).startswith('distance_grouped') for i in itemset)

# 6. Apriori: extraer reglas age_qcut → distance_grouped
print("\n=== Apriori edad→distancia ===")
support_values    = [0.01,0.005,0.001]
confidence_values = [0.3,0.2]
lift_thresholds   = [1.0,1.5]
for sup in support_values:
    print(f"\n-- min_support = {sup*100:.2f}% --")
    freq = apriori(transactions, min_support=sup, use_colnames=True)
    if freq.empty:
        print("No hay itemsets frecuentes.")
        continue
    for conf in confidence_values:
        print(f"  Confianza ≥ {conf*100:.0f}%:")
        rules = association_rules(freq, metric='confidence', min_threshold=conf)
        mask = rules['antecedents'].apply(is_age) & rules['consequents'].apply(is_distance)
        sel = rules[mask]
        if sel.empty:
            print("    Sin reglas edad→distancia.")
            continue
        for lt in lift_thresholds:
            subset = sel[sel['lift']>=lt]
            if subset.empty:
                print(f"    Ninguna con lift ≥ {lt}.")
            else:
                print(f"    Reglas con lift ≥ {lt}:" )
                print(
                    subset
                    .sort_values('lift',ascending=False)
                    .loc[:,['antecedents','consequents','support','confidence','lift']]
                    .head(5)
                )

# 7. Asociación con FP-Growth
# ---------------------------
# Pruebas de FP-Growth con umbrales más bajos
support_values   = [0.001, 0.0005, 0.0001]
confidence_values= [0.2, 0.15, 0.1]
lift_thresholds  = [1.0, 1.2]

for sup in support_values:
    print(f"\n--- FP-Growth min_support = {sup*100:.3f}% ---")
    freq_fp = fpgrowth(transactions, min_support=sup, use_colnames=True)
    if freq_fp.empty:
        print("No frecuentes a este soporte.")
        continue
    for conf in confidence_values:
        print(f"  > min_confidence = {conf*100:.0f}%")
        rules_fp = association_rules(
            freq_fp,
            metric='confidence',
            min_threshold=conf
        )
        # Filtrar edad → ritmo
        mask_fp = (
            rules_fp['antecedents'].apply(is_age) &
            rules_fp['consequents'].apply(is_distance)
        )
        sel_fp = rules_fp[mask_fp]
        if sel_fp.empty:
            print("    * Sin reglas edad→distancia.")
            continue
        # Mostrar reglas según lift uno o dos umbrales
        for lt in lift_thresholds:
            subset = sel_fp[sel_fp['lift']>=lt]
            if subset.empty:
                print(f"    - Ninguna con lift ≥ {lt}.")
            else:
                print(f"    - Reglas con lift ≥ {lt}:")
                print(subset
                      .sort_values('lift', ascending=False)
                      .loc[:, ['antecedents','consequents','support','confidence','lift']]
                      .head(5))


print("\nProceso completo.")
