# -*- coding: utf-8 -*-
"""
main_prism.py
--------------
Punto de entrada para ejecutar el algoritmo PRISM sobre el dataset
de proyectos de software (predicción de Exito_Proyecto), siguiendo el
pseudocódigo visto en clase (documento "Reglas_PRISM"):

    PROCEDIMIENTO Recubrimiento_secuencial(Clases, atributos, ejemplos)
    PROCEDIMIENTO AprenderUnaRegla(Clase, Ejemplos, Atributos)
    mejorRestriccion(Restricciones, regla)

Uso:
    python main_prism.py

Parámetros configurables abajo en la sección CONFIGURACIÓN.
"""

import os
import pandas as pd

from datos import cargar_dataset, preparar_datos, resumen_dataset
from algoritmo_prism import ejecutar_prism, ordenar_mejores_reglas
from graficos import generar_todas_las_graficas

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
RUTA_DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_prism_proyectos_software.csv")
COLUMNA_OBJETIVO = "Exito_Proyecto"
COLUMNA_ID = "Proyecto_ID"

# Precisión objetivo durante la CONSTRUCCIÓN de cada regla. El algoritmo
# clásico de PRISM (Cendrowska, 1987) exige 1.0: el bucle "AprenderUnaRegla"
# sigue añadiendo condiciones MIENTRAS la regla cubra algún ejemplo negativo,
# es decir, hasta que la regla sea 100% pura (o se queden sin atributos).
PRECISION_OBJETIVO_CONSTRUCCION = 1.0

# Umbrales mínimos para ACEPTAR una regla en el conjunto FINAL (filtro de
# calidad posterior a la construcción), según lo indicado en clase:
#   COBERTURA_MINIMA  == "soporte" mínimo (proporción del dataset que cubre la regla)
#   PRECISION_MINIMA  == "confianza" mínima (qué tan pura es la regla para su clase)
COBERTURA_MINIMA = 0.70
PRECISION_MINIMA = 0.85

CRITERIO_ORDEN = "precision"   # 'precision', 'cobertura' o 'n_cubiertas'
TOP_N_REGLAS = 15
RUTA_SALIDA_CSV = os.path.join(os.path.dirname(__file__), "reglas_prism_resultado.csv")


def main():
    print("=" * 70)
    print("ALGORITMO PRISM - Reglas de clasificación (Proyectos de Software)")
    print("=" * 70)

    # 1. Cargar y explorar el dataset
    df = cargar_dataset(RUTA_DATASET)
    resumen = resumen_dataset(df, columna_objetivo=COLUMNA_OBJETIVO)
    print(f"\nInstancias (proyectos): {resumen['n_instancias']}")
    print("Distribución de la clase objetivo:")
    print(resumen["distribucion_clases"].to_string())
    print("\nAtributos disponibles y sus valores posibles:")
    for atributo, valores in resumen["valores_por_atributo"].items():
        print(f"  - {atributo}: {valores}")

    # 2. Preparar instancias (lista de diccionarios)
    instancias, atributos = preparar_datos(df, columna_objetivo=COLUMNA_OBJETIVO, columna_id=COLUMNA_ID)

    # 3. Ejecutar PRISM (Recubrimiento_secuencial)
    print(f"\nEjecutando PRISM (precision_objetivo_construccion="
          f"{PRECISION_OBJETIVO_CONSTRUCCION} -> regla 'pura' clásica) ...")
    print(f"Filtro final de calidad: cobertura_minima={COBERTURA_MINIMA}, "
          f"precision_minima={PRECISION_MINIMA}")

    reglas_sin_filtrar = ejecutar_prism(
        instancias,
        atributos,
        columna_objetivo=COLUMNA_OBJETIVO,
        precision_objetivo=PRECISION_OBJETIVO_CONSTRUCCION,
        cobertura_minima=0.0,
        precision_minima=0.0,
    )
    print(f"Reglas generadas por el algoritmo (sin filtrar): {len(reglas_sin_filtrar)}")

    reglas = [
        r for r in reglas_sin_filtrar
        if r.cobertura >= COBERTURA_MINIMA and r.precision >= PRECISION_MINIMA
    ]
    print(f"Reglas que superan el filtro final (cobertura>={COBERTURA_MINIMA}, "
          f"precision>={PRECISION_MINIMA}): {len(reglas)}")

    if not reglas:
        print("\n[Aviso] Ninguna regla individual alcanzó una cobertura tan alta "
              f"({COBERTURA_MINIMA}). Esto es normal: una sola regla con pocas "
              "condiciones rara vez cubre el 70% de un dataset con varios "
              "atributos categóricos. Mostrando en su lugar las mejores reglas "
              "SIN el filtro de cobertura mínima (sólo con precision_minima):")
        reglas = [r for r in reglas_sin_filtrar if r.precision >= PRECISION_MINIMA]
        print(f"Reglas con precision >= {PRECISION_MINIMA} (cualquier cobertura): {len(reglas)}")

    # 4. Obtener las mejores reglas según el criterio elegido
    mejores_reglas = ordenar_mejores_reglas(reglas, criterio=CRITERIO_ORDEN, top_n=TOP_N_REGLAS)

    print(f"\nTop {min(TOP_N_REGLAS, len(mejores_reglas))} mejores reglas por '{CRITERIO_ORDEN}':")
    print("-" * 70)
    for i, regla in enumerate(mejores_reglas, start=1):
        d = regla.a_diccionario()
        print(f"{i:2d}. {regla.texto()}")
        print(f"     Confianza (Fracción)={d['Confianza (Fracción)']}  "
              f"Confianza={d['Confianza']}  "
              f"Cobertura={d['Cobertura']}  "
              f"Soporte (Fracción)={d['Soporte (Fracción)']}  "
              f"Soporte={d['Soporte']}  "
              f"Lift={d['Lift']}")

    # 5. Guardar todas las reglas (sin filtrar) en un CSV, marcando cuáles pasan el filtro
    if reglas_sin_filtrar:
        filas = []
        for r in sorted(reglas_sin_filtrar, key=lambda r: r.precision, reverse=True):
            d = r.a_diccionario()
            d["Pasa Filtro Final"] = (r.cobertura >= COBERTURA_MINIMA and r.precision >= PRECISION_MINIMA)
            filas.append(d)
        df_reglas = pd.DataFrame(filas)
        df_reglas.to_csv(RUTA_SALIDA_CSV, index=False, encoding="utf-8-sig")
        print(f"\nTodas las reglas generadas ({len(reglas_sin_filtrar)}) guardadas en: {RUTA_SALIDA_CSV}")
    else:
        print("\nNo se generaron reglas.")

    # 6. Generar gráficas (con el conjunto de reglas ya filtrado que se reporta arriba)
    generar_todas_las_graficas(resumen["distribucion_clases"], reglas)

    print("\nProceso de PRISM finalizado con éxito.")


if __name__ == "__main__":
    main()
