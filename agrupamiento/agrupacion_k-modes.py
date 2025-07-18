#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from kmodes.kmodes import KModes
import argparse


def kmodes_cluster(input_csv: str,
                    k: int,
                    sample_n: int,
                    seed: int,
                    output_csv: str):
    """
    Carga un CSV, toma una muestra reproducible y realiza clustering categórico con K-Modes
    usando las columnas categóricas definidas.

    - input_csv: ruta al CSV de entrada
    - k: número de clusters
    - sample_n: tamaño de la muestra de filas a procesar
    - seed: semilla para muestreo y K-Modes
    - output_csv: ruta donde guardar el CSV con la columna 'cluster'
    """
    # Leer datos completos
    df = pd.read_csv(input_csv)
    # Normalizar nombres de columnas
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(' ', '_')
    )
    
    # Muestreo reproducible
    df_sample = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    # Columnas categóricas para clustering
    cat_cols = [
        'grouped_distances',
        #'athlete_country',
        'athlete_gender',
        'grouped_pace',
        'age_without_gen'
    ]
    data = df_sample[cat_cols].astype(str)

    # Configurar y ajustar K-Modes
    km = KModes(
        n_clusters=k,
        init='Cao',
        n_init=5,
        verbose=1,
        random_state=seed
    )
    clusters = km.fit_predict(data)

    # Asignar cluster y guardar resultado
    df_sample['cluster'] = clusters
    df_sample.to_csv(output_csv, index=False)
    print(f"Clusters guardados en '{output_csv}'")

    # Mostrar tamaño de cada cluster
    print("\nTamaño de cada cluster:")
    print(df_sample['cluster'].value_counts().sort_index().to_string())

    # Mostrar distribución de cada variable por cluster
    for col in cat_cols:
        print(f"\nDistribución de '{col}' por cluster:")
        ct = pd.crosstab(df_sample['cluster'], df_sample[col], normalize='index').round(2)
        print(ct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Clustering categórico con K-Modes sobre muestra de datos"
    )
    parser.add_argument(
        'input_csv', nargs='?', default='datos_preprocesados_v6.csv',
        help='CSV de entrada (por defecto datos_preprocesados_v6.csv)'
    )
    parser.add_argument(
        '--k', type=int, default=5,
        help='Número de clusters (por defecto 5)'
    )
    parser.add_argument(
        '--sample', type=int, default=10000,
        help='Tamaño de la muestra de filas (por defecto 10000)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Semilla para muestreo y K-Modes (por defecto 42)'
    )
    parser.add_argument(
        '--output_csv', default='clusters_kmodes.csv',
        help='CSV de salida con la columna cluster (por defecto clusters_kmodes.csv)'
    )
    args = parser.parse_args()

    kmodes_cluster(
        input_csv=args.input_csv,
        k=args.k,
        sample_n=args.sample,
        seed=args.seed,
        output_csv=args.output_csv
    )
