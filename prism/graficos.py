# -*- coding: utf-8 -*-
"""
graficos.py
------------
Funciones de visualización para los resultados del algoritmo PRISM.
Genera y guarda las gráficas en la carpeta 'graficas/'.
"""

from __future__ import annotations
import os
from typing import List
import matplotlib.pyplot as plt

from algoritmo_prism import ReglaPRISM

CARPETA_GRAFICAS = os.path.join(os.path.dirname(__file__), "graficas")
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

plt.rcParams["figure.dpi"] = 110


def _guardar(fig, nombre_archivo: str):
    ruta = os.path.join(CARPETA_GRAFICAS, nombre_archivo)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Gráfica guardada en: {ruta}")


def graficar_distribucion_clases(distribucion_clases):
    """Gráfica de barras con la distribución de la clase objetivo."""
    fig, ax = plt.subplots(figsize=(5, 4))
    colores = ["#55A868", "#C44E52"]
    ax.bar(distribucion_clases.index.astype(str), distribucion_clases.values,
           color=colores[: len(distribucion_clases)])
    ax.set_ylabel("Número de proyectos")
    ax.set_title("Distribución de la clase objetivo (Exito_Proyecto)")
    for i, v in enumerate(distribucion_clases.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    _guardar(fig, "01_distribucion_clases.png")


def graficar_reglas_por_clase(reglas: List[ReglaPRISM]):
    """Cuántas reglas se generaron para cada valor de la clase."""
    if not reglas:
        return
    conteo = {}
    for r in reglas:
        conteo[r.clase] = conteo.get(r.clase, 0) + 1

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(conteo.keys()), list(conteo.values()), color="#4C72B0")
    ax.set_ylabel("Cantidad de reglas")
    ax.set_title("Reglas PRISM generadas por clase")
    for i, (k, v) in enumerate(conteo.items()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    _guardar(fig, "02_reglas_por_clase.png")


def graficar_top_reglas(reglas: List[ReglaPRISM], top_n: int = 10, criterio: str = "precision"):
    """Gráfica de barras horizontales con las mejores reglas según
    precisión (confianza) o cobertura (soporte)."""
    if not reglas:
        print("  (sin reglas para graficar)")
        return

    reglas_ordenadas = sorted(reglas, key=lambda r: getattr(r, criterio), reverse=True)[:top_n]
    etiquetas = [
        (" Y ".join(f"{a}={v}" for a, v in r.condiciones) or "(vacía)") + f"  ->  {r.clase}"
        for r in reglas_ordenadas
    ]
    valores = [getattr(r, criterio) for r in reglas_ordenadas]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(etiquetas))))
    ax.barh(etiquetas[::-1], valores[::-1], color="#8172B2")
    ax.set_xlabel(criterio.capitalize())
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Top {top_n} reglas PRISM por {criterio}")
    _guardar(fig, f"03_top_reglas_{criterio}.png")


def graficar_dispersion_cobertura_precision(reglas: List[ReglaPRISM]):
    """Scatter: cobertura (soporte) vs precisión (confianza), coloreado por clase."""
    if not reglas:
        return
    fig, ax = plt.subplots(figsize=(7, 6))

    clases = sorted({r.clase for r in reglas})
    colores = {clase: color for clase, color in zip(clases, ["#55A868", "#C44E52", "#4C72B0", "#DD8452"])}

    for clase in clases:
        subset = [r for r in reglas if r.clase == clase]
        xs = [r.cobertura for r in subset]
        ys = [r.precision for r in subset]
        tam = [max(30, r.n_cubiertas * 8) for r in subset]
        ax.scatter(xs, ys, s=tam, alpha=0.75, edgecolor="k", label=f"Clase = {clase}",
                   color=colores.get(clase))

    ax.set_xlabel("Cobertura (soporte)")
    ax.set_ylabel("Precisión (confianza)")
    ax.set_title("Reglas PRISM: Cobertura vs Precisión")
    ax.legend()
    _guardar(fig, "04_dispersion_cobertura_precision.png")


def graficar_numero_condiciones(reglas: List[ReglaPRISM]):
    """Histograma de cuántas condiciones (atributo=valor) tiene cada regla."""
    if not reglas:
        return
    n_condiciones = [len(r.condiciones) for r in reglas]
    fig, ax = plt.subplots(figsize=(5, 4))
    valores_unicos = sorted(set(n_condiciones))
    conteos = [n_condiciones.count(v) for v in valores_unicos]
    ax.bar([str(v) for v in valores_unicos], conteos, color="#DD8452")
    ax.set_xlabel("Número de condiciones en la regla")
    ax.set_ylabel("Cantidad de reglas")
    ax.set_title("Complejidad de las reglas generadas por PRISM")
    _guardar(fig, "05_numero_condiciones.png")


def generar_todas_las_graficas(distribucion_clases, reglas):
    """Función de conveniencia que genera todas las gráficas del módulo."""
    print("Generando gráficas de PRISM...")
    graficar_distribucion_clases(distribucion_clases)
    graficar_reglas_por_clase(reglas)
    graficar_top_reglas(reglas, criterio="precision")
    graficar_top_reglas(reglas, criterio="cobertura")
    graficar_dispersion_cobertura_precision(reglas)
    graficar_numero_condiciones(reglas)
