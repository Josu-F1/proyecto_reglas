# -*- coding: utf-8 -*-
"""
datos.py
--------
Carga y preprocesamiento del dataset de proyectos de software para
el algoritmo PRISM (inducción de reglas de clasificación).

El dataset tiene columnas categóricas (Tamano_Equipo, Metodologia,
Presupuesto, Experiencia_Lider) y una columna objetivo (Exito_Proyecto)
con valores 'Si' / 'No'.
"""

from __future__ import annotations
import pandas as pd
from typing import List, Dict, Tuple


def cargar_dataset(ruta_csv: str) -> pd.DataFrame:
    """Carga el CSV crudo tal cual está en disco."""
    df = pd.read_csv(ruta_csv)
    return df


def preparar_datos(
    df: pd.DataFrame,
    columna_objetivo: str = "Exito_Proyecto",
    columna_id: str = "Proyecto_ID",
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Convierte el DataFrame en una lista de "instancias" (diccionarios
    atributo -> valor), separando la columna identificadora (si existe)
    y dejando la columna objetivo dentro de cada instancia.

    Devuelve:
        instancias: lista de dicts {atributo: valor, ..., objetivo: valor}
        atributos: lista de nombres de columnas de atributos (sin el ID ni el objetivo)
    """
    columnas = [c for c in df.columns if c not in (columna_id, columna_objetivo)]
    instancias = []
    for _, fila in df.iterrows():
        instancia = {col: fila[col] for col in columnas}
        instancia[columna_objetivo] = fila[columna_objetivo]
        instancias.append(instancia)
    return instancias, columnas


def resumen_dataset(df: pd.DataFrame, columna_objetivo: str = "Exito_Proyecto") -> dict:
    """Pequeño resumen del dataset: tamaño, distribución de clases,
    valores posibles de cada atributo."""
    distribucion_clases = df[columna_objetivo].value_counts()
    columnas_atributos = [c for c in df.columns if c not in (columna_objetivo, "Proyecto_ID")]
    valores_por_atributo = {col: sorted(df[col].unique().tolist()) for col in columnas_atributos}

    return {
        "n_instancias": len(df),
        "distribucion_clases": distribucion_clases,
        "atributos": columnas_atributos,
        "valores_por_atributo": valores_por_atributo,
    }


if __name__ == "__main__":
    df = cargar_dataset("../data/dataset_prism_proyectos_software.csv")
    resumen = resumen_dataset(df)
    print("Instancias:", resumen["n_instancias"])
    print("Distribución de clases:")
    print(resumen["distribucion_clases"])
    print("Atributos y valores posibles:")
    for atributo, valores in resumen["valores_por_atributo"].items():
        print(f"  {atributo}: {valores}")
