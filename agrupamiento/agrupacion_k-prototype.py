#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
import argparse

def cluster_and_plot_kproto(input_csv: str,
                            k: int,
                            sample_n: int,
                            seed: int,
                            output_csv: str = None):
    # 1) Cargo todo el CSV
    df = pd.read_csv(input_csv)
    
    # 2) Normalizo nombres de columnas: lower-case, quito espacios → guiones bajos
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(' ', '_')
                 )
    
    # 3) Columnas a usar tras normalizar:
    num_cols = ['pace_min_per_km', 'athlete_age']
    cat_cols = ['athlete_gender','grouped_distances'] #'age_category_revised', 'athlete_country' quiete esta por que es la combinacion de age vs genero
    
    # 4) Muestreo para la visualización
    df_sample = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    
    # 5) Escalo las numéricas
    scaler = StandardScaler()
    num_data = scaler.fit_transform(df_sample[num_cols])
    
    # 6) Preparo las categóricas
    cat_data = df_sample[cat_cols].astype(str).to_numpy()
    
    # 7) Concateno num + cat para K‑Prototypes
    X = np.hstack([num_data, cat_data])
    
    # 8) Índices de variables categóricas en X
    cat_idx = list(range(num_data.shape[1], num_data.shape[1] + len(cat_cols)))
    
    # 9) Ajusto K‑Prototypes
    kproto = KPrototypes(
        n_clusters=k,
        init='Cao',
        verbose=1,
        random_state=seed
    )
    clusters = kproto.fit_predict(X, categorical=cat_idx)
    df_sample['cluster'] = clusters
    
    # 10) (Opcional) Guardar asignaciones
    if output_csv:
        df_sample.to_csv(output_csv, index=False)
        print(f"Asignaciones de cluster guardadas en '{output_csv}'")

    # 9) Guardar si se solicita
    if output_csv:
        df_sample.to_csv(output_csv, index=False)
        print(f"Asignaciones de cluster guardadas en '{output_csv}'")

    # 10) Generar tabla resumen por cluster
    summary = df_sample.groupby('cluster').apply(lambda g: pd.Series({
        'size': len(g),
        '%_total': len(g) / len(df_sample) * 100,
        'mean_distance': g['grouped_distances'].mean(),
        'mean_pace': g['pace_min_per_km'].mean(),
        'mean_age': g['athlete_age'].mean(),
        '%_male': (g['athlete_gender'] == 'M').mean() * 100,
    })).round(2)
    print("\nResumen de clusters:\n")
    print(summary)

    
    # 11a) Gráfico Edad vs Ritmo, coloreado por cluster
    plt.figure(figsize=(8,6))
    plt.scatter(
        df_sample['athlete_age'],
        df_sample['pace_min_per_km'],
        c=df_sample['cluster'],
        s=10,
        alpha=0.7
    )
    plt.xlabel('Athlete age')
    plt.ylabel('Pace (min per km)')
    plt.title(f'K‑Prototypes Clusters (k={k}) — Edad vs Ritmo')
    plt.tight_layout()
    plt.show()
    
    # 11b) Gráfico Edad vs Distancia, coloreado por cluster
    plt.figure(figsize=(8,6))
    plt.scatter(
        df_sample['athlete_age'],
        df_sample['distance_km'],
        c=df_sample['cluster'],
        s=10,
        alpha=0.7
    )
    plt.xlabel('Athlete age')
    plt.ylabel('Distance (km)')
    plt.title(f'K‑Prototypes Clusters (k={k}) — Edad vs Distancia')
    plt.tight_layout()
    plt.show()
    
    # 12) Tamaño de cada cluster
    print("\nTamaño de cada cluster:")
    print(df_sample['cluster'].value_counts().sort_index().to_string())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Clustering mixto con K‑Prototypes y gráficos Edad vs Ritmo/Distancia"
    )
    parser.add_argument(
        'input_csv',
        nargs='?',
        default='datos_preprocesados_v5.csv',
        help='CSV de entrada (por defecto datos_preprocesados_v5.csv)'
    )
    parser.add_argument(
        '--k', type=int, default=4,
        help='Número de clusters k (por defecto 4)'
    )
    parser.add_argument(
        '--sample', type=int, default=5000,
        help='Tamaño de la muestra para graficar (por defecto 100000)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Semilla para el muestreo y K‑Prototypes (por defecto 42)'
    )
    parser.add_argument(
        '--output_csv', default=None,
        help='CSV donde guardar las filas muestreadas con su cluster (opcional)'
    )
    args = parser.parse_args()
    
    cluster_and_plot_kproto(
        input_csv=args.input_csv,
        k=args.k,
        sample_n=args.sample,
        seed=args.seed,
        output_csv=args.output_csv
    )
