import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# 1. Carga de datos
df = pd.read_csv('datos_preprocesados.csv')

# 2. Discretización de variables
# 2.1 Edad en 5 quintiles
age_labels = ['Q1','Q2','Q3','Q4','Q5']
df['age_qbin'], age_bins = pd.qcut(
    df['Athlete age'],
    q=5,
    labels=age_labels,
    retbins=True
)
print("Edades quintiles:", age_bins)
print(df['age_qbin'].value_counts().sort_index(), "\n")

# 2.2 Ritmo en 4 cuartiles (bins fijos basados en cuantiles)
pace_labels = ['pace_Q1','pace_Q2','pace_Q3','pace_Q4']
df['pace_qbin'], pace_bins = pd.qcut(
    df['Pace_min_per_km'],
    q=4,
    labels=pace_labels,
    retbins=True
)
print("Pace cuartiles:", pace_bins)
print(df['pace_qbin'].value_counts().sort_index(), "\n")

# 2.3 Distancia en bins definidos
dist_bins = [50, 100, 200, 300, df['Distance_km'].max()]
dist_labels = ['50-100km','100-200km','200-300km','300+km']
df['dist_bin'] = pd.cut(
    df['Distance_km'],
    bins=dist_bins,
    labels=dist_labels,
    include_lowest=True
)
print("Distancias:")
print(df['dist_bin'].value_counts().sort_index(), "\n")

# 3. Generar matriz transaccional (one-hot)
one_hot = pd.get_dummies(
    df[['age_qbin','pace_qbin','dist_bin']],
    prefix_sep='_'
).clip(upper=1)

# 4. Apriori
print("=== Apriori ===")
support=0.005    # 0.5%
confidence=0.3  # 30%
freq_ap = apriori(one_hot, min_support=support, use_colnames=True)
rules_ap = association_rules(freq_ap, metric='confidence', min_threshold=confidence)
print(rules_ap[['antecedents','consequents','support','confidence','lift']].sort_values('lift',ascending=False).head(10), "\n")

# 5. FP-Growth
print("=== FP-Growth ===")
min_sup=0.001   # 0.1%
freq_fp = fpgrowth(one_hot, min_support=min_sup, use_colnames=True)
rules_fp = association_rules(freq_fp, metric='confidence', min_threshold=confidence)
print(rules_fp[['antecedents','consequents','support','confidence','lift']].sort_values('lift',ascending=False).head(10))
