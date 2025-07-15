#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse

def add_pace_and_age_cats(input_csv: str,
                          output_csv: str = 'datos_preprocesados_v6.csv'):
    """
    - Carga un CSV con las columnas 'pace_min_per_km' y 'age_category_revised'.
    - Añade dos columnas:
        * grouped_pace: categorías de ritmo basadas en pace_min_per_km
        * age_without_gen: extrae sólo los dígitos de age_category_revised (quita M/F)
    - Guarda el resultado en output_csv.
    """
    # 1) Leer
    df = pd.read_csv(input_csv)

    # 2) Normalizar nombres de columna
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(' ', '_')
    )

    # 3) Crear grouped_pace
    bins = [0, 6, 8, 10, 12, 14, 16, 18, 20, np.inf]
    labels = ['<6','6-8','8-10','10-12','12-14','14-16','16-18','18-20','>20']
    df['grouped_pace'] = pd.cut(
        df['pace_min_per_km'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # 4) Crear age_without_gen
    #    Quita la primera letra (M o F) de age_category_revised
    df['age_without_gen'] = df['age_category_revised'].astype(str).str[1:]

    # 5) Guardar CSV resultante
    df.to_csv(output_csv, index=False)
    print(f"Nuevo CSV guardado en '{output_csv}' con columnas 'grouped_pace' y 'age_without_gen'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Añade grouped_pace y age_without_gen a un CSV existente"
    )
    parser.add_argument(
        'input_csv',
        nargs='?',
        default='datos_preprocesados_v5.csv',
        help='CSV de entrada con pace_min_per_km y age_category_revised (por defecto datos_preprocesados_v5.csv)'
    )
    parser.add_argument(
        '--output_csv',
        default='datos_preprocesados_v6.csv',
        help='Ruta de salida para el nuevo CSV (por defecto datos_preprocesados_v6.csv)'
    )
    args = parser.parse_args()

    add_pace_and_age_cats(args.input_csv, args.output_csv)
