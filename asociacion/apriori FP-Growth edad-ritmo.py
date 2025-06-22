import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# 1. Carga y preprocesamiento de datos
# ------------------------------------
df = pd.read_csv('datos_preprocesados.csv')

# 2. Discretización de edad: varias opciones
# -------------------------------------------
# 2.1. Por cuantiles (grupos de igual frecuencia)
df['age_qcut'] = pd.qcut(
    df['Athlete age'],
    q=4,
    labels=['Q1_<25pct','Q2_25-50pct','Q3_50-75pct','Q4_75-100pct']
)
print("Conteo por age_qcut (cuartiles de igual frecuencia):")
print(df['age_qcut'].value_counts().sort_index(), "\n")

# 2.2. Por percentiles definidos (20,40,60,80)
cuts = df['Athlete age'].quantile([0.2,0.4,0.6,0.8]).tolist()
bins = [df['Athlete age'].min()] + cuts + [df['Athlete age'].max()]
labels = ['P0-20','P20-40','P40-60','P60-80','P80-100']
df['age_pct_bin'] = pd.cut(
    df['Athlete age'],
    bins=bins,
    labels=labels,
    include_lowest=True
)
print("Conteo por age_pct_bin (percentiles 20/40/60/80):")
print(df['age_pct_bin'].value_counts().sort_index(), "\n")

# 2.3. K-means univariante sobre edad (centroides como cortes)
from sklearn.cluster import KMeans
ages = df[['Athlete age']].values
km = KMeans(n_clusters=5, random_state=0).fit(ages)
centroids = sorted(km.cluster_centers_.flatten())
boundaries = [df['Athlete age'].min()] + centroids + [df['Athlete age'].max()]
labels_km = [f'K{i}' for i in range(1, len(boundaries))]
df['age_km_bin'] = pd.cut(
    df['Athlete age'],
    bins=boundaries,
    labels=labels_km,
    include_lowest=True
)
print("Conteo por age_km_bin (KMeans univariante):")
print(df['age_km_bin'].value_counts().sort_index(), "\n")

# 3. Discretización de ritmos por cuartiles (qcut)
# -----------------------------------------------
df['pace_qcat'] = pd.qcut(
    df['Pace_min_per_km'],
    q=4,
    labels=['pace_<6.82','pace_6.82-8.18','pace_8.18-10.23','pace_>10.23']
)
print("Conteo por pace_qcat (cuartiles de ritmo):")
print(df['pace_qcat'].value_counts().sort_index(), "\n")

# 4. Preparar transacciones y funciones auxiliares
# ------------------------------------------------
# Elige uno de los age bins para el análisis, p.ej. age_qcut.
once_hot = pd.get_dummies(df[['age_qcut','pace_qcat']]).clip(upper=1)
transactions = once_hot.copy()

def is_age(itemset):
    return any(str(i).startswith('age_qcut') for i in itemset)

def is_pace(itemset):
    return any(str(i).startswith('pace_qcat') for i in itemset)

# 5. Asociación con Apriori
# -------------------------
support_values    = [0.01,0.005,0.001]
confidence_values = [0.3,0.2]
lift_thresholds   = [1.0,1.5]
for sup in support_values:
    print(f"\n=== Apriori soporte={sup} ===")
    freq = apriori(transactions, min_support=sup, use_colnames=True)
    for conf in confidence_values:
        rules = association_rules(freq, metric='confidence', min_threshold=conf)
        mask = rules['antecedents'].apply(is_age) & rules['consequents'].apply(is_pace)
        sel = rules[mask]
        if not sel.empty:
            for lt in lift_thresholds:
                top = sel[sel['lift']>=lt]
                if not top.empty:
                    print(f"-- conf={conf}, lift>={lt}")
                    print(top[['antecedents','consequents','support','confidence','lift']].head(),"\n")

# 6. Asociación con FP-Growth
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
            rules_fp['consequents'].apply(is_pace)
        )
        sel_fp = rules_fp[mask_fp]
        if sel_fp.empty:
            print("    * Sin reglas edad→pace.")
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