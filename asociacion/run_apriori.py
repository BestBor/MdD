#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import argparse


def run_apriori(input_csv: str,
                min_support: float,
                min_confidence: float,
                top_n_country: int,
                top_n_club: int,
                output_rules_csv: str = None):
    # 1) Carga y normaliza columnas
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # 2) Reducir cardinalidad de country y club
    for col, top_n in [('athlete_country', top_n_country),
                       ('athlete_club', top_n_club)]:
        top_vals = df[col].value_counts().nlargest(top_n).index
        df[col] = df[col].where(df[col].isin(top_vals), other='Other')
    
    # 3) Seleccionar variables categóricas
    cat_cols = [
        'grouped_distances',
        'grouped_pace',
        'age_without_gen',
        'cluster',
        'athlete_country',
        'athlete_club',
        'athlete_gender'
    ]
    df_cat = df[cat_cols].astype(str)
    
    # 4) One-hot encoding
    df_encoded = pd.get_dummies(df_cat, prefix_sep='=')
    
    # 5) Generar itemsets frecuentes
    frequent_itemsets = apriori(df_encoded,
                                min_support=min_support,
                                use_colnames=True)
    
    # 6) Generar reglas de asociación
    rules = association_rules(frequent_itemsets,
                              metric='confidence',
                              min_threshold=min_confidence)
    
    # 7) Mostrar top 10 reglas por lift
    if rules.empty:
        print("No se encontraron reglas con esos parámetros.")
    else:
        rules_sorted = rules.sort_values(by='lift', ascending=False)
        print("\nTop 10 reglas por lift:\n")
        for _, row in rules_sorted.head(20).iterrows():
            antecedents = ','.join(row['antecedents'])
            consequents = ','.join(row['consequents'])
            print(f"{antecedents} -> {consequents} "
                  f"(support={row['support']:.3f}, "
                  f"confidence={row['confidence']:.2f}, "
                  f"lift={row['lift']:.2f})")
        
        # 7b) Reglas dedicadas a cada cluster
        unique_clusters = sorted(df['cluster'].astype(str).unique())
        for cluster_id in unique_clusters:
            target = f'cluster={cluster_id}'
            rules_for = rules[rules['consequents']
                               .apply(lambda cs: target in cs)]
            if rules_for.empty:
                print(f"\nNo hay reglas para {target}.")
                continue
            rules_for = rules_for.sort_values(by='lift', ascending=False).head(10)
            print(f"\n=== Reglas para {target} (top 5 por lift) ===")
            for _, r in rules_for.iterrows():
                ants = ', '.join(r['antecedents'])
                cons = ', '.join(r['consequents'])
                print(f"{ants} -> {cons}  "
                      f"(support={r['support']:.3f}, "
                      f"confidence={r['confidence']:.2f}, "
                      f"lift={r['lift']:.2f})")
    
    # 8) Guardar reglas a CSV
    if output_rules_csv:
        rules.to_csv(output_rules_csv, index=False)
        print(f"\nReglas completas guardadas en '{output_rules_csv}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Apriori sobre variables categóricas de ultras con segmentación por cluster"
    )
    parser.add_argument('input_csv',
                        help='CSV con columnas categóricas procesadas e incl. cluster')
    parser.add_argument('--min_support',
                        type=float, default=0.01,
                        help='Soporte mínimo (por defecto 0.01)')
    parser.add_argument('--min_confidence',
                        type=float, default=0.3,
                        help='Confianza mínima (por defecto 0.3)')
    parser.add_argument('--top_n_country',
                        type=int, default=10,
                        help='Número de países más frecuentes (por defecto 10)')
    parser.add_argument('--top_n_club',
                        type=int, default=10,
                        help='Número de clubes más frecuentes (por defecto 10)')
    parser.add_argument('--output_rules_csv',
                        default=None,
                        help='CSV donde guardar todas las reglas (opcional)')
    args = parser.parse_args()
    
    run_apriori(args.input_csv,
                args.min_support,
                args.min_confidence,
                args.top_n_country,
                args.top_n_club,
                args.output_rules_csv)
