# -*- coding: utf-8 -*-
"""
datos.py
--------
Carga y preprocesamiento del dataset transaccional (supermercado)
para el algoritmo Apriori.

El dataset viene en formato "binario ancho":
    Transaccion_ID, Leche, Pan, Huevos, ...
    1, 1, 1, 0, ...

IMPORTANTE (alineado con el material de referencia "Extracción de Reglas
de Asociación: El Algoritmo Apriori"): Apriori aquí NO se limita a mirar
únicamente los productos comprados (valor = 1). Cada columna se trata
como un **atributo** que puede tomar el valor 1 (comprado) o 0 (no
comprado), y el algoritmo genera ítems de la forma "Atributo=Valor" para
AMBOS estados, exactamente como en el ejemplo de la 'cesta de la compra'
del documento (leche=1, leche=0, queso=1, queso=0, ...).

Esto convierte cada transacción en un conjunto de pares atributo-valor:
    {'Leche=1', 'Pan=1', 'Huevos=0', 'Mantequilla=0', ...}
que es el formato general que espera el algoritmo Apriori para poder
descubrir reglas de asociación entre CUALQUIER combinación de pares
atributo-valor (no solo entre productos comprados).
"""

from __future__ import annotations
import pandas as pd
from typing import List, Set


def cargar_dataset(ruta_csv: str) -> pd.DataFrame:
    """Carga el CSV crudo tal cual está en disco."""
    df = pd.read_csv(ruta_csv)
    return df


def obtener_columnas_productos(df: pd.DataFrame, columna_id: str = "Transaccion_ID") -> List[str]:
    """Devuelve el nombre de todas las columnas que representan productos
    (es decir, todas menos la columna identificadora)."""
    return [c for c in df.columns if c != columna_id]


def convertir_a_transacciones(df: pd.DataFrame, columna_id: str = "Transaccion_ID") -> List[Set[str]]:
    """Convierte el DataFrame binario ancho en una lista de sets de pares
    "Atributo=Valor", incluyendo TANTO el valor 1 (producto comprado)
    COMO el valor 0 (producto no comprado) de cada columna.

    Ejemplo para la fila (Leche=1, Pan=1, Huevos=0, ...):
        {'Leche=1', 'Pan=1', 'Huevos=0', ...}

    Esto permite que Apriori descubra reglas del tipo
    "SI Huevos=0 Y Mantequilla=0 ENTONCES Queso=0" (ausencias que se
    repiten juntas), tal como se ilustra en el documento de referencia,
    y no únicamente reglas de "productos comprados juntos".
    """
    columnas_productos = obtener_columnas_productos(df, columna_id)
    transacciones: List[Set[str]] = []

    for _, fila in df.iterrows():
        items_fila = {f"{col}={int(fila[col])}" for col in columnas_productos}
        transacciones.append(items_fila)

    return transacciones


def convertir_a_transacciones_solo_comprados(df: pd.DataFrame, columna_id: str = "Transaccion_ID") -> List[Set[str]]:
    """Variante clásica de 'canasta de mercado': cada transacción es sólo
    el conjunto de productos con valor 1 (comprados), SIN los estados en 0.
    Se deja disponible por si se desea comparar contra la versión
    generalizada (atributo=valor) que usa main_apriori.py por defecto.
    """
    columnas_productos = obtener_columnas_productos(df, columna_id)
    transacciones: List[Set[str]] = []
    for _, fila in df.iterrows():
        comprados = {producto for producto in columnas_productos if fila[producto] == 1}
        transacciones.append(comprados)
    return transacciones


def resumen_dataset(df: pd.DataFrame, columna_id: str = "Transaccion_ID") -> dict:
    """Genera un pequeño resumen estadístico del dataset, útil para
    mostrar en consola o en un reporte antes de correr Apriori."""
    columnas_productos = obtener_columnas_productos(df, columna_id)
    n_transacciones = len(df)
    frecuencia_productos = df[columnas_productos].sum().sort_values(ascending=False)
    tamano_promedio_canasta = df[columnas_productos].sum(axis=1).mean()

    return {
        "n_transacciones": n_transacciones,
        "n_productos": len(columnas_productos),
        "productos": columnas_productos,
        "frecuencia_productos": frecuencia_productos,
        "tamano_promedio_canasta": round(tamano_promedio_canasta, 2),
    }


if __name__ == "__main__":
    # Pequeña prueba manual del módulo
    df = cargar_dataset("../data/dataset_apriori_supermercado.csv")
    resumen = resumen_dataset(df)
    print("Transacciones:", resumen["n_transacciones"])
    print("Productos:", resumen["n_productos"])
    print("Tamaño promedio de canasta:", resumen["tamano_promedio_canasta"])
    print(resumen["frecuencia_productos"])
