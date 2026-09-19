# -*- coding: utf-8 -*-
"""
graficos.py
------------
Funciones de visualización para los resultados del algoritmo Apriori.
Genera y guarda las gráficas en la carpeta 'graficas/'.
"""

from __future__ import annotations
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

from algoritmo_apriori import ReglaAsociacion

CARPETA_GRAFICAS = os.path.join(os.path.dirname(__file__), "graficas")
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

plt.rcParams["figure.dpi"] = 110


def _guardar(fig, nombre_archivo: str):
    ruta = os.path.join(CARPETA_GRAFICAS, nombre_archivo)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Gráfica guardada en: {ruta}")


def graficar_frecuencia_productos(frecuencia_productos: pd.Series, top_n: int = 10):
    """Gráfica de barras con los productos más comprados (valor=1) en el
    dataset (esto es sólo exploratorio; Apriori en sí trabaja con
    pares Atributo=Valor para ambos estados, 1 y 0)."""
    datos = frecuencia_productos.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(datos.index[::-1], datos.values[::-1], color="#4C72B0")
    ax.set_xlabel("Número de transacciones")
    ax.set_title(f"Top {top_n} productos más comprados (valor = 1)")
    for i, v in enumerate(datos.values[::-1]):
        ax.text(v, i, f" {v}", va="center")
    _guardar(fig, "01_frecuencia_productos.png")


def graficar_top_reglas(reglas: List[ReglaAsociacion], top_n: int = 10, criterio: str = "lift"):
    """Gráfica de barras horizontales con las mejores reglas según lift/confianza."""
    if not reglas:
        print("  (sin reglas para graficar)")
        return

    reglas_ordenadas = sorted(reglas, key=lambda r: getattr(r, criterio), reverse=True)[:top_n]
    etiquetas = [
        f"{', '.join(sorted(r.antecedente))} -> {', '.join(sorted(r.consecuente))}"
        for r in reglas_ordenadas
    ]
    valores = [getattr(r, criterio) for r in reglas_ordenadas]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(etiquetas))))
    ax.barh(etiquetas[::-1], valores[::-1], color="#55A868")
    ax.set_xlabel(criterio.capitalize())
    ax.set_title(f"Top {top_n} reglas de asociación por {criterio}")
    _guardar(fig, f"02_top_reglas_{criterio}.png")


def graficar_dispersion_soporte_confianza(reglas: List[ReglaAsociacion]):
    """Scatter plot: soporte vs confianza, tamaño de punto según lift."""
    if not reglas:
        return
    soportes = [r.soporte for r in reglas]
    confianzas = [r.confianza for r in reglas]
    lifts = [r.lift for r in reglas]

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(soportes, confianzas, s=[max(20, l * 40) for l in lifts],
                          c=lifts, cmap="viridis", alpha=0.75, edgecolor="k")
    ax.set_xlabel("Soporte")
    ax.set_ylabel("Confianza")
    ax.set_title("Reglas de asociación: Soporte vs Confianza (color/tamaño = Lift)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Lift")
    _guardar(fig, "03_dispersion_soporte_confianza.png")


def graficar_cantidad_itemsets_por_nivel(itemsets_por_nivel: dict):
    """Gráfica de barras: cuántos itemsets frecuentes se hallaron por tamaño."""
    if not itemsets_por_nivel:
        return
    niveles = sorted(itemsets_por_nivel.keys())
    cantidades = [len(itemsets_por_nivel[n]) for n in niveles]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Tamaño {n}" for n in niveles], cantidades, color="#C44E52")
    ax.set_ylabel("Cantidad de itemsets frecuentes")
    ax.set_title("Itemsets frecuentes encontrados por nivel (Apriori)")
    for i, v in enumerate(cantidades):
        ax.text(i, v, str(v), ha="center", va="bottom")
    _guardar(fig, "04_itemsets_por_nivel.png")


def generar_todas_las_graficas(frecuencia_productos, itemsets_por_nivel, reglas):
    """Función de conveniencia que genera todas las gráficas del módulo."""
    print("Generando gráficas de Apriori...")
    graficar_frecuencia_productos(frecuencia_productos)
    graficar_cantidad_itemsets_por_nivel(itemsets_por_nivel)
    graficar_top_reglas(reglas, criterio="lift")
    graficar_top_reglas(reglas, criterio="confianza")
    graficar_dispersion_soporte_confianza(reglas)
