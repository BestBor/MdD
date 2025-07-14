#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def group_distances(input_csv: str,
                    output_csv: str) -> pd.DataFrame:
    """
    Lee el CSV de entrada, agrupa Distance_km en categorías,
    añade la columna 'grouped_distances' y guarda en output_csv.
    Devuelve el DataFrame resultante.
    """
    # 1) Carga el dataset
    df = pd.read_csv(input_csv)

    # 2) Define los bins y sus etiquetas
    max_dist = df['Distance_km'].max()
    bins   = [0, 70, 100, 160, 200, max_dist]
    labels = ['≤70', '71–100', '101–160', '161–200', '>200']

    # 3) Crea la columna agrupada
    df['grouped_distances'] = pd.cut(
        df['Distance_km'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # 4) Guarda el CSV resultante
    df.to_csv(output_csv, index=False)
    print(f"Archivo guardado en '{output_csv}'")

    # 5) Imprime el recuento por categoría
    counts = df['grouped_distances'].value_counts().sort_index()
    print("\nRecuento por grouped_distances:")
    print(counts.to_string())

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Agrupa Distance_km en categorías y guarda en un nuevo CSV."
    )
    parser.add_argument(
        'input_csv',
        nargs='?',
        default='datos_preprocesados_v4.csv',
        help='CSV de entrada con la columna Distance_km (por defecto datos_preprocesados_v4.csv)'
    )
    parser.add_argument(
        '--output_csv',
        default='datos_preprocesados_v5.csv',
        help='CSV de salida con la nueva columna grouped_distances (por defecto datos_preprocesados_v5.csv)'
    )
    args = parser.parse_args()

    group_distances(args.input_csv, args.output_csv)

if __name__ == '__main__':
    main()
