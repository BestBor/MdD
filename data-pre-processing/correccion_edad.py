#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def assign_age_category(age: float, gender: str) -> str:
    """
    Dado un age (numérico) y gender ('m' o 'f'),
    retorna la categoría revisada, p.ej. 'M25' o 'F30'.
    """
    # Prefijo según género
    g = str(gender).strip().upper()
    prefix = 'M' if g == 'M' else 'F'
    
    # Rangos de edad
    if age < 20:
        bucket = '20'
    elif age <= 25:
        bucket = '25'
    elif age <= 30:
        bucket = '30'
    elif age <= 35:
        bucket = '35'
    elif age <= 40:
        bucket = '40'
    elif age <= 45:
        bucket = '45'
    elif age <= 50:
        bucket = '50'
    elif age <= 55:
        bucket = '55'
    elif age <= 60:
        bucket = '60'
    elif age <= 65:
        bucket = '65'
    elif age <= 70:
        bucket = '70'
    elif age <= 75:
        bucket = '75'
    elif age <= 80:
        bucket = '80'
    else:
        # Si hay edades > 80, las agrupamos en '80'
        bucket = '80'
    
    return f"{prefix}{bucket}"

def main():
    parser = argparse.ArgumentParser(
        description="Añade la columna age_category_revised según edad y género."
    )
    parser.add_argument(
        'input_csv',
        nargs='?',
        default='datos_preprocesados_v3.csv',
        help='CSV de entrada con las columnas Athlete age y Athlete gender'
    )
    parser.add_argument(
        '--output_csv',
        default='datos_preprocesados_v4.csv',
        help='CSV de salida donde se guardará la nueva columna'
    )
    args = parser.parse_args()

    # 1) Leer el dataset completo
    df = pd.read_csv(args.input_csv)

    # 2) Crear la columna revisada
    df['age_category_revised'] = df.apply(
        lambda row: assign_age_category(row['Athlete age'], row['Athlete gender']),
        axis=1
    )

    # 3) Guardar el resultado
    df.to_csv(args.output_csv, index=False)
    print(f"Nuevo CSV guardado en '{args.output_csv}' con la columna 'age_category_revised'.")

    # 4) (Opcional) Verificar conteo por categoría revisada
    counts = df['age_category_revised'].value_counts().sort_index()
    print("\nConteo por age_category_revised:")
    print(counts.to_string())

if __name__ == '__main__':
    main()
