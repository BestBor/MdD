#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def unify_age_categories(path,
                         column='Athlete age category',
                         output_path='datos_preprocesados_v3.csv'):
    """
    - Carga el CSV completo.
    - En la columna `column`, elimina la 'U' de categorías tipo 'MU20' → 'M20', 'WU23' → 'W23', etc.
    - Guarda el DataFrame modificado en `output_path`.
    - Devuelve un DataFrame resumen con estadísticas por categoría unificada.
    """
    # 1) Leer todo el dataset
    df = pd.read_csv(path)

    # 2) Unificar categorías: MUxx → Mxx, WUxx → Wxx
    df[column] = df[column].astype(str).str.replace(
        r'^([MW])U(\d+)$',  # letra M o W seguida de U y dígitos
        r'\1\2',            # conserva la letra y los dígitos
        regex=True
    )

    # 3) Guardar el dataset modificado
    df.to_csv(output_path, index=False)
    print(f"Dataset con categorías unificadas guardado en '{output_path}'")

    # 4) Recalcular estadísticas por categoría
    counts = df[column].value_counts()
    avg_speed   = df.groupby(column)['Speed_calc_kmh'].mean()
    median_pace = df.groupby(column)['Pace_min_per_km'].median()

    summary = (
        pd.DataFrame({
            column: counts.index,
            'age_cat_size': counts.values,
            'age_cat_avg_speed': [avg_speed[c] for c in counts.index],
            'age_cat_median_pace': [median_pace[c] for c in counts.index]
        })
        .sort_values(column)
        .reset_index(drop=True)
    )
    return summary

def main():
    parser = argparse.ArgumentParser(
        description="Unifica categorías 'U' en Athlete age category y recalcula stats."
    )
    parser.add_argument(
        'input_csv',
        nargs='?',
        default='datos_preprocesados_v2.csv',
        help='CSV de entrada (por defecto datos_preprocesados_v2.csv)'
    )
    parser.add_argument(
        '--output_csv',
        default='datos_preprocesados_v3.csv',
        help='CSV de salida con categorías unificadas (por defecto datos_preprocesados_v3.csv)'
    )
    args = parser.parse_args()

    summary_df = unify_age_categories(
        path=args.input_csv,
        output_path=args.output_csv
    )

    # Mostrar el resumen en consola
    pd.set_option('display.max_rows', None)
    print("\nResumen por categoría unificada:\n")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
