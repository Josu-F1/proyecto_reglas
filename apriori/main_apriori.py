# -*- coding: utf-8 -*-
"""
main_apriori.py
-----------------
Punto de entrada para ejecutar el algoritmo Apriori sobre el dataset
de transacciones del supermercado, siguiendo el mismo procedimiento del
documento de referencia "Extracción de Reglas de Asociación: El
Algoritmo Apriori" (Fase 0: cobertura mínima; Fase 1: itemsets
frecuentes; Fase 2: generación de reglas).

Uso:
    python main_apriori.py

Parámetros configurables abajo en la sección CONFIGURACIÓN.
"""

import os
import pandas as pd

from datos import cargar_dataset, convertir_a_transacciones, resumen_dataset
from algoritmo_apriori import ejecutar_apriori, ordenar_mejores_reglas, calcular_cobertura_minima
from graficos import generar_todas_las_graficas

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
RUTA_DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_apriori_supermercado.csv")

# Valores por defecto tomados del ejemplo del documento de referencia
# (Parámetros del Sistema: Soporte Mínimo = 0.67, Confianza Mínima = 0.80)
SOPORTE_MINIMO = 0.70
CONFIANZA_MINIMA = 0.85

CRITERIO_ORDEN = "lift"    # 'lift', 'confianza' o 'soporte'
TOP_N_REGLAS = 15
RUTA_SALIDA_CSV = os.path.join(os.path.dirname(__file__), "reglas_apriori_resultado.csv")


def main():
    print("=" * 70)
    print("ALGORITMO APRIORI - Reglas de asociación (Dataset Supermercado)")
    print("=" * 70)

    # 1. Cargar y explorar el dataset
    df = cargar_dataset(RUTA_DATASET)
    resumen = resumen_dataset(df)
    print(f"\nTransacciones: {resumen['n_transacciones']}")
    print(f"Productos distintos: {resumen['n_productos']}")
    print(f"Tamaño promedio de canasta: {resumen['tamano_promedio_canasta']} productos")
    print("\nProductos más comprados (valor=1):")
    print(resumen["frecuencia_productos"].head(5).to_string())

    # 2. Convertir a formato de transacciones GENERALIZADO: cada transacción
    #    es un conjunto de pares "Atributo=Valor" (incluye tanto los
    #    productos comprados, valor=1, como los NO comprados, valor=0),
    #    tal como en el documento de referencia (leche=1, leche=0, ...).
    transacciones = convertir_a_transacciones(df)

    # --- Fase 0: Determinando la Cobertura Mínima ---
    cobertura_minima = calcular_cobertura_minima(len(transacciones), SOPORTE_MINIMO)
    print(f"\n--- Fase 0: Determinando la Cobertura Mínima ---")
    print(f"Registros totales en BD = {len(transacciones)}")
    print(f"Soporte mínimo requerido = {SOPORTE_MINIMO}")
    print(f"Cobertura Mínima = round({len(transacciones)} x {SOPORTE_MINIMO}) = {cobertura_minima}")
    print("Para que un itemset se considere suficientemente frecuente, el número de "
          f"registros que lo cumplen debe ser de al menos {cobertura_minima}.")

    # 3. Ejecutar Apriori: Fase 1 (itemsets frecuentes) + Fase 2 (reglas)
    print(f"\nEjecutando Apriori con soporte_minimo={SOPORTE_MINIMO} "
          f"y confianza_minima={CONFIANZA_MINIMA} ...")
    itemsets_por_nivel, reglas = ejecutar_apriori(
        transacciones,
        soporte_minimo=SOPORTE_MINIMO,
        confianza_minima=CONFIANZA_MINIMA,
    )

    total_itemsets = sum(len(v) for v in itemsets_por_nivel.values())
    print(f"\n--- Fase 1: Itemsets frecuentes encontrados: {total_itemsets} ---")
    for nivel, itemsets in itemsets_por_nivel.items():
        print(f"  - Tamaño {nivel}: {len(itemsets)} itemsets")
        if nivel == 1:
            for itemset, cobertura in sorted(itemsets.items(), key=lambda x: -x[1]):
                item_txt = list(itemset)[0]
                print(f"      {item_txt}  (cobertura={cobertura})")

    print(f"\n--- Fase 2: Reglas generadas (confianza >= {CONFIANZA_MINIMA}): {len(reglas)} ---")

    # 4. Obtener las mejores reglas según el criterio elegido
    mejores_reglas = ordenar_mejores_reglas(reglas, criterio=CRITERIO_ORDEN, top_n=TOP_N_REGLAS)

    print(f"\nTop {min(TOP_N_REGLAS, len(mejores_reglas))} mejores reglas por '{CRITERIO_ORDEN}':")
    print("-" * 70)
    for i, regla in enumerate(mejores_reglas, start=1):
        d = regla.a_diccionario()
        print(f"{i:2d}. {regla.texto()}")
        print(f"     cobertura={d['cobertura']}/{d['cobertura_antecedente']}  "
              f"soporte={d['soporte']}  confianza={d['confianza']}  "
              f"lift={d['lift']}")

    # 5. Guardar todas las reglas en un CSV
    if reglas:
        filas = [r.a_diccionario() for r in sorted(reglas, key=lambda r: r.lift, reverse=True)]
        df_reglas = pd.DataFrame(filas)
        df_reglas.to_csv(RUTA_SALIDA_CSV, index=False, encoding="utf-8-sig")
        print(f"\nTodas las reglas ({len(reglas)}) guardadas en: {RUTA_SALIDA_CSV}")
    else:
        print("\nNo se generaron reglas con los parámetros actuales. "
              "Prueba bajando SOPORTE_MINIMO o CONFIANZA_MINIMA.")

    # 6. Generar gráficas
    generar_todas_las_graficas(resumen["frecuencia_productos"], itemsets_por_nivel, reglas)

    print("\nProceso de Apriori finalizado con éxito.")


if __name__ == "__main__":
    main()
