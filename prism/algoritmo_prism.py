# -*- coding: utf-8 -*-
"""
algoritmo_prism.py
--------------------
Implementación "desde cero" del algoritmo PRISM para
inducción de reglas de clasificación del tipo:

    SI (Atributo1 = Valor1) Y (Atributo2 = Valor2) ... ENTONCES Clase = X

A diferencia de un árbol de decisión, PRISM construye las reglas
directamente, una por una, usando una estrategia de "cobertura"
(covering algorithm):

Para cada valor de clase objetivo (ej. 'Si', 'No'):
    Mientras existan instancias de esa clase aún no cubiertas por
    ninguna regla:
        1. Empezar una regla vacía (que cubre TODAS las instancias
           restantes).
        2. Repetir:
             - Probar añadir, a la regla, cada condición posible
               (Atributo = Valor) que todavía no esté en la regla.
             - Para cada condición candidata, calcular:
                    p = instancias restantes que cumplen la regla+condición
                        y pertenecen a la clase objetivo
                    t = instancias restantes que cumplen la regla+condición
                        (sin importar la clase)
                    precision = p / t
             - Elegir la condición con mayor precisión (y en caso de
               empate, la que cubra más instancias -> mayor t).
             - Añadir esa condición a la regla.
           Hasta que la regla sea "perfecta" (precision = 1.0, es decir,
           sólo cubre instancias de la clase objetivo) o ya no queden
           atributos para agregar.
        3. Guardar la regla. Eliminar del conjunto de trabajo las
           instancias de la clase objetivo que la regla cubrió
           correctamente (esto es lo que hace que el algoritmo avance).

Además de la versión clásica, esta implementación permite fijar un
umbral mínimo de "cobertura" (equivalente al soporte) y de "precisión"
(equivalente a la confianza) para aceptar una regla, lo que resulta útil
en datasets ruidosos donde exigir precisión = 1.0 generaría reglas
demasiado específicas (sobreajustadas).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class ReglaPRISM:
    """Representa una regla de clasificación inducida por PRISM."""
    condiciones: List[Tuple[str, Any]]   # [(atributo, valor), ...]
    clase: str
    cobertura: float          # proporción de instancias del dataset ORIGINAL que cumplen la regla (soporte)
    precision: float          # proporción de esas instancias que SÍ son de la clase (confianza)
    n_cubiertas: int          # cuántas instancias del dataset original cumplen la regla
    n_correctas: int          # cuántas de esas son realmente de la clase
    n_total: int = 1000       # total de instancias en el dataset
    n_clase_total: int = 500  # total de instancias de la clase objetivo en el dataset

    @property
    def lift(self) -> float:
        soporte_clase = self.n_clase_total / self.n_total if self.n_total else 1.0
        return self.precision / soporte_clase if soporte_clase > 0 else 0.0

    @property
    def condicion_restriccion(self) -> str:
        partes = " Y ".join(f"{attr} = {val}" for attr, val in self.condiciones)
        return partes if partes else "(sin condiciones)"

    def texto(self) -> str:
        return f"SI {self.condicion_restriccion}  ENTONCES  Exito_Proyecto = {self.clase}"

    def a_diccionario(self) -> dict:
        return {
            "Condición (Restricción)": self.condicion_restriccion,
            "Confianza (Fracción)": f"{self.n_correctas}/{self.n_cubiertas}" if self.n_cubiertas else "0/0",
            "Confianza": round(self.precision, 4),
            "Cobertura": self.n_cubiertas,
            "Soporte (Fracción)": f"{self.n_cubiertas}/{self.n_total}",
            "Soporte": round(self.cobertura, 4),
            "Lift": round(self.lift, 4),
            "Clase": self.clase,
            "Regla Completa": self.texto(),
        }

    def cumple(self, instancia: Dict[str, Any]) -> bool:
        """Indica si una instancia cumple TODAS las condiciones de la regla."""
        return all(instancia.get(attr) == val for attr, val in self.condiciones)


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def _valores_posibles_por_atributo(
    instancias: List[Dict[str, Any]], atributos: List[str]
) -> Dict[str, List[Any]]:
    """Obtiene todos los valores posibles que puede tomar cada atributo."""
    valores: Dict[str, set] = {attr: set() for attr in atributos}
    for inst in instancias:
        for attr in atributos:
            valores[attr].add(inst[attr])
    return {attr: sorted(vals) for attr, vals in valores.items()}


def _filtrar_por_condiciones(
    instancias: List[Dict[str, Any]], condiciones: List[Tuple[str, Any]]
) -> List[Dict[str, Any]]:
    """Devuelve sólo las instancias que cumplen TODAS las condiciones dadas."""
    resultado = instancias
    for attr, val in condiciones:
        resultado = [inst for inst in resultado if inst[attr] == val]
    return resultado


def _evaluar_regla_en_dataset_completo(
    dataset_completo: List[Dict[str, Any]],
    condiciones: List[Tuple[str, Any]],
    columna_objetivo: str,
    clase_objetivo: str,
) -> Tuple[int, int, float, float]:
    """Evalúa una regla (lista de condiciones) contra el dataset ORIGINAL
    completo (no el subconjunto de trabajo), para reportar métricas reales
    de cobertura (soporte) y precisión (confianza)."""
    cubiertas = _filtrar_por_condiciones(dataset_completo, condiciones)
    n_cubiertas = len(cubiertas)
    n_correctas = sum(1 for inst in cubiertas if inst[columna_objetivo] == clase_objetivo)
    cobertura = n_cubiertas / len(dataset_completo) if dataset_completo else 0.0
    precision = n_correctas / n_cubiertas if n_cubiertas else 0.0
    return n_cubiertas, n_correctas, cobertura, precision


# ---------------------------------------------------------------------------
# Algoritmo PRISM principal
# ---------------------------------------------------------------------------

def _construir_una_regla(
    instancias_trabajo: List[Dict[str, Any]],
    dataset_completo: List[Dict[str, Any]],
    atributos: List[str],
    columna_objetivo: str,
    clase_objetivo: str,
    precision_objetivo: float = 1.0,
) -> ReglaPRISM:
    """Construye UNA regla para la clase_objetivo, añadiendo condiciones
    una por una hasta alcanzar la precisión objetivo (o quedarse sin
    atributos posibles)."""

    condiciones: List[Tuple[str, Any]] = []
    atributos_disponibles = list(atributos)
    subconjunto_actual = list(instancias_trabajo)

    while atributos_disponibles:
        mejor_condicion = None
        mejor_precision = -1.0
        mejor_p = -1

        for attr in atributos_disponibles:
            valores_posibles = sorted({inst[attr] for inst in subconjunto_actual})
            for valor in valores_posibles:
                cubiertas = [inst for inst in subconjunto_actual if inst[attr] == valor]
                t = len(cubiertas)
                if t == 0:
                    continue
                p = sum(1 for inst in cubiertas if inst[columna_objetivo] == clase_objetivo)
                precision_candidata = p / t

                # Criterio de selección: mayor precisión: en caso de empate, mayor cobertura (t)
                if (precision_candidata > mejor_precision) or (
                    precision_candidata == mejor_precision and t > mejor_p
                ):
                    mejor_precision = precision_candidata
                    mejor_p = t
                    mejor_condicion = (attr, valor)

        if mejor_condicion is None:
            break

        condiciones.append(mejor_condicion)
        attr_elegido = mejor_condicion[0]
        atributos_disponibles.remove(attr_elegido)
        subconjunto_actual = [inst for inst in subconjunto_actual if inst[attr_elegido] == mejor_condicion[1]]

        if mejor_precision >= precision_objetivo:
            break
        if len(subconjunto_actual) == 0:
            break

    n_cubiertas, n_correctas, cobertura, precision = _evaluar_regla_en_dataset_completo(
        dataset_completo, condiciones, columna_objetivo, clase_objetivo
    )
    n_total = len(dataset_completo)
    n_clase_total = sum(1 for inst in dataset_completo if inst[columna_objetivo] == clase_objetivo)

    return ReglaPRISM(
        condiciones=condiciones,
        clase=clase_objetivo,
        cobertura=cobertura,
        precision=precision,
        n_cubiertas=n_cubiertas,
        n_correctas=n_correctas,
        n_total=n_total,
        n_clase_total=n_clase_total,
    )


def prism_para_una_clase(
    instancias: List[Dict[str, Any]],
    atributos: List[str],
    columna_objetivo: str,
    clase_objetivo: str,
    precision_objetivo: float = 1.0,
    max_reglas: int = 30,
) -> List[ReglaPRISM]:
    """Ejecuta el algoritmo de cobertura PRISM para UNA clase objetivo,
    generando reglas hasta cubrir todas las instancias de esa clase
    (o alcanzar max_reglas como salvaguarda contra loops infinitos)."""

    reglas: List[ReglaPRISM] = []
    instancias_trabajo = list(instancias)  # se irán "quitando" instancias cubiertas

    instancias_clase_restantes = [
        inst for inst in instancias_trabajo if inst[columna_objetivo] == clase_objetivo
    ]

    while instancias_clase_restantes and len(reglas) < max_reglas:
        regla = _construir_una_regla(
            instancias_trabajo,
            instancias,  # dataset completo original, para métricas reales
            atributos,
            columna_objetivo,
            clase_objetivo,
            precision_objetivo=precision_objetivo,
        )

        if not regla.condiciones:
            # No se pudo aislar ninguna condición útil -> evitar loop infinito
            break

        reglas.append(regla)

        # Quitar del conjunto de trabajo las instancias de la clase objetivo
        # que esta regla cubrió CORRECTAMENTE (ya están "explicadas")
        cubiertas_por_regla = _filtrar_por_condiciones(instancias_trabajo, regla.condiciones)
        ids_cubiertas_correctas = [
            id(inst) for inst in cubiertas_por_regla if inst[columna_objetivo] == clase_objetivo
        ]
        instancias_trabajo = [
            inst for inst in instancias_trabajo if id(inst) not in ids_cubiertas_correctas
        ]
        instancias_clase_restantes = [
            inst for inst in instancias_trabajo if inst[columna_objetivo] == clase_objetivo
        ]

    return reglas


def ejecutar_prism(
    instancias: List[Dict[str, Any]],
    atributos: List[str],
    columna_objetivo: str = "Exito_Proyecto",
    precision_objetivo: float = 1.0,
    cobertura_minima: float = 0.0,
    precision_minima: float = 0.0,
) -> List[ReglaPRISM]:
    """Ejecuta PRISM para TODOS los valores posibles de la clase objetivo
    y devuelve la lista completa de reglas, filtradas opcionalmente por
    cobertura mínima (soporte) y precisión mínima (confianza).
    """
    clases = sorted({inst[columna_objetivo] for inst in instancias})
    todas_las_reglas: List[ReglaPRISM] = []

    for clase in clases:
        reglas_clase = prism_para_una_clase(
            instancias,
            atributos,
            columna_objetivo,
            clase,
            precision_objetivo=precision_objetivo,
        )
        todas_las_reglas.extend(reglas_clase)

    # Filtrado final por umbrales de soporte/confianza (si se especificaron)
    reglas_filtradas = [
        r for r in todas_las_reglas
        if r.cobertura >= cobertura_minima and r.precision >= precision_minima
    ]

    return reglas_filtradas


# ---------------------------------------------------------------------------
# Alias con los nombres EXACTOS del pseudocódigo visto en clase
# (documento "Reglas_PRISM"), para que el mapeo teoría <-> código sea directo:
#
#   PROCEDIMIENTO Recubrimiento_secuencial(Clases, atributos, ejemplos)
#   PROCEDIMIENTO AprenderUnaRegla(Clase, Ejemplos, Atributos)
#   mejorRestriccion(Restricciones, regla)  -> la condición con mayor
#       confianza (precision); en caso de empate, la de mayor cobertura.
# ---------------------------------------------------------------------------

def mejorRestriccion(subconjunto_actual, atributos_disponibles, columna_objetivo, clase_objetivo):
    """Equivalente directo de 'mejorRestriccion(Restricciones, regla)' del
    pseudocódigo: evalúa cada condición candidata (Atributo=Valor) posible
    sobre el subconjunto actual de instancias y devuelve la de mayor
    confianza (precision = p/t), y en caso de empate la de mayor cobertura (t).
    Devuelve (atributo, valor, precision, cobertura) o None si no hay candidatos.
    """
    mejor_condicion = None
    mejor_precision = -1.0
    mejor_t = -1

    for attr in atributos_disponibles:
        valores_posibles = sorted({inst[attr] for inst in subconjunto_actual})
        for valor in valores_posibles:
            cubiertas = [inst for inst in subconjunto_actual if inst[attr] == valor]
            t = len(cubiertas)
            if t == 0:
                continue
            p = sum(1 for inst in cubiertas if inst[columna_objetivo] == clase_objetivo)
            precision_candidata = p / t
            if (precision_candidata > mejor_precision) or (
                precision_candidata == mejor_precision and t > mejor_t
            ):
                mejor_precision = precision_candidata
                mejor_t = t
                mejor_condicion = (attr, valor)

    if mejor_condicion is None:
        return None
    return (mejor_condicion[0], mejor_condicion[1], mejor_precision, mejor_t)


# AprenderUnaRegla == _construir_una_regla ; Recubrimiento_secuencial == ejecutar_prism
AprenderUnaRegla = _construir_una_regla
Recubrimiento_secuencial = ejecutar_prism


def ordenar_mejores_reglas(
    reglas: List[ReglaPRISM],
    criterio: str = "precision",
    top_n: int = 10,
) -> List[ReglaPRISM]:
    """Ordena las reglas de mayor a menor según el criterio indicado
    ('precision', 'cobertura', 'n_cubiertas') y devuelve las top_n mejores."""
    clave = {
        "precision": lambda r: (r.precision, r.cobertura),
        "cobertura": lambda r: (r.cobertura, r.precision),
        "n_cubiertas": lambda r: r.n_cubiertas,
    }.get(criterio, lambda r: (r.precision, r.cobertura))

    return sorted(reglas, key=clave, reverse=True)[:top_n]
