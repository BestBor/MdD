#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import argparse


def compare_and_plot(proto_csv: str,
                     modes_csv: str,
                     sample_n: int,
                     seed: int,
                     output_csv: str = None):
    """
    Carga dos CSVs con resultados de cluster (K-Prototypes y K-Modes), los fusiona,
    calcula métricas de comparación y genera scatter plots de Edad vs Ritmo coloreados
    por cada tipo de cluster.

    - proto_csv: ruta al CSV con cluster de K-Prototypes
    - modes_csv: ruta al CSV con cluster de K-Modes
    - sample_n: número de puntos a muestrear para graficar
    - seed: semilla para muestreo
    - output_csv: ruta para guardar CSV fusionado con ambas etiquetas
    """
    # 1) Leer archivos
    df_p = pd.read_csv(proto_csv)
    df_m = pd.read_csv(modes_csv)

    # 2) Normalizar nombres de columnas
    df_p.columns = df_p.columns.str.strip().str.lower().str.replace(' ', '_')
    df_m.columns = df_m.columns.str.strip().str.lower().str.replace(' ', '_')

    # 3) Renombrar columnas de cluster
    df_p = df_p.rename(columns={'cluster': 'cluster_proto'})
    df_m = df_m.rename(columns={'cluster': 'cluster_modes'})

    # 4) Merge por athlete_id
    df = pd.merge(
        df_p[['athlete_id', 'athlete_age', 'pace_min_per_km', 'cluster_proto']],
        df_m[['athlete_id', 'cluster_modes']],
        on='athlete_id',
        how='inner'
    )

    # 5) Calcular Adjusted Rand Index
    ari = adjusted_rand_score(df['cluster_proto'], df['cluster_modes'])
    print(f"Adjusted Rand Index: {ari:.4f}\n")

    # 6) Tabla de contingencia
    print("Tabla de contingencia (proto vs modes):\n")
    ct = pd.crosstab(df['cluster_proto'], df['cluster_modes'], normalize='index').round(2)
    print(ct, "\n")

    # 7) Guardar merge si se solicita
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Merged CSV guardado en '{output_csv}'\n")

    # 8) Muestreo para graficar
    df_plot = df.sample(n=sample_n, random_state=seed)

    # 9) Scatter Edad vs Ritmo por cluster_proto y cluster_modes
    fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)
    sc0 = axes[0].scatter(
        df_plot['athlete_age'], df_plot['pace_min_per_km'],
        c=df_plot['cluster_proto'], cmap='tab10', s=10, alpha=0.7
    )
    axes[0].set_title('K-Prototypes Clusters')
    axes[0].set_xlabel('Athlete age')
    axes[0].set_ylabel('Pace (min per km)')

    sc1 = axes[1].scatter(
        df_plot['athlete_age'], df_plot['pace_min_per_km'],
        c=df_plot['cluster_modes'], cmap='tab10', s=10, alpha=0.7
    )
    axes[1].set_title('K-Modes Clusters')
    axes[1].set_xlabel('Athlete age')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compara clusters de K-Prototypes y K-Modes con gráficos"
    )
    parser.add_argument('proto_csv',
                        help='CSV con clusters de K-Prototypes')
    parser.add_argument('modes_csv',
                        help='CSV con clusters de K-Modes')
    parser.add_argument('--sample', type=int, default=20000,
                        help='Número de puntos a graficar')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla para muestreo')
    parser.add_argument('--output_csv', default=None,
                        help='Ruta para guardar CSV fusionado')
    args = parser.parse_args()

    compare_and_plot(
        proto_csv=args.proto_csv,
        modes_csv=args.modes_csv,
        sample_n=args.sample,
        seed=args.seed,
        output_csv=args.output_csv
    )
