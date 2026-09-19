# -*- coding: utf-8 -*-
"""
algoritmo_apriori.py
---------------------
Implementación del algoritmo Apriori para minería de reglas de asociación:

    Fase 0 - Determinar la Cobertura Mínima:
        cobertura_minima = round(n_transacciones * soporte_minimo)


    Fase 1 - Generación de itemsets frecuentes (iterativo, k=1,2,3,...):
        1. Generar itemsets de tamaño 1 (todos los pares Atributo=Valor).
        2. Filtrar por cobertura >= cobertura_minima.
        3. Combinar (unir) los itemsets frecuentes de tamaño k para
           generar candidatos de tamaño k+1, y podar los candidatos cuyo
           subconjunto de tamaño k no sea frecuente (propiedad Apriori).
        4. Repetir hasta que no se generen más itemsets frecuentes.

    Fase 2 - Generación de reglas (Anatomía de la Confianza):
        A partir de cada itemset frecuente de tamaño >= 2, se generan
        TODAS las reglas antecedente -> consecuente posibles (antecedente
        único, antecedente doble, etc. -> "explosión combinatoria"), y se
        conservan las que cumplan la confianza mínima:

            confianza(A -> B) = cobertura(A U B) / cobertura(A)

        Además se calculan métricas adicionales: soporte, lift, leverage
        y conviction.
"""

from __future__ import annotations
from itertools import combinations
from typing import Dict, FrozenSet, List, Set, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class ReglaAsociacion:
    """Representa una regla de asociación: antecedente -> consecuente."""
    antecedente: FrozenSet[str]
    consecuente: FrozenSet[str]
    cobertura: int            # nº de transacciones que cumplen antecedente Y consecuente (conteo absoluto)
    cobertura_antecedente: int  # nº de transacciones que cumplen solo el antecedente
    soporte: float             # soporte del itemset completo (antecedente U consecuente) = cobertura / n
    confianza: float           # P(consecuente | antecedente) = cobertura / cobertura_antecedente
    lift: float                 # confianza / soporte(consecuente)

    def texto(self) -> str:
        ant = " AND ".join(sorted(self.antecedente))
        con = " AND ".join(sorted(self.consecuente))
        return f"SI {ant}  ENTONCES  {con}"

    def a_diccionario(self) -> dict:
        return {
            "antecedente": " AND ".join(sorted(self.antecedente)),
            "consecuente": " AND ".join(sorted(self.consecuente)),
            "cobertura": self.cobertura,
            "cobertura_antecedente": self.cobertura_antecedente,
            "soporte": round(self.soporte, 4),
            "confianza": round(self.confianza, 4),
            "lift": round(self.lift, 4),
        }


# ---------------------------------------------------------------------------
# Fase 0: traducir el soporte mínimo a una cobertura mínima (conteo absoluto)
# ---------------------------------------------------------------------------

def calcular_cobertura_minima(n_transacciones: int, soporte_minimo: float) -> int:
    return round(n_transacciones * soporte_minimo)


# ---------------------------------------------------------------------------
# Fase 1: generación de itemsets frecuentes
# ---------------------------------------------------------------------------

def _contar_cobertura(transacciones: List[Set[str]], itemset: FrozenSet[str]) -> int:
    """Cobertura de un itemset: número ABSOLUTO de transacciones que
    contienen TODOS los items del itemset (antecedente y consecuente)."""
    return sum(1 for t in transacciones if itemset.issubset(t))


def _generar_candidatos_iniciales(transacciones: List[Set[str]]) -> List[FrozenSet[str]]:
    """Genera todos los itemsets de tamaño 1 a partir de los productos
    presentes en las transacciones."""
    items_unicos = set()
    for t in transacciones:
        items_unicos.update(t)
    return [frozenset([item]) for item in sorted(items_unicos)]


def _generar_candidatos_siguientes(itemsets_frecuentes_k: List[FrozenSet[str]], k: int) -> List[FrozenSet[str]]:
    """A partir de los itemsets frecuentes de tamaño k, genera candidatos
    de tamaño k+1, uniendo pares que comparten k-1 elementos (método
    estándar de Apriori) y luego podando los que tengan algún subconjunto
    de tamaño k que NO sea frecuente.
    """
    candidatos: Set[FrozenSet[str]] = set()
    lista = list(itemsets_frecuentes_k)
    n = len(lista)

    for i in range(n):
        for j in range(i + 1, n):
            union = lista[i] | lista[j]
            if len(union) == k + 1:
                candidatos.add(union)

    # Poda: todo subconjunto de tamaño k del candidato debe ser frecuente
    frecuentes_set = set(itemsets_frecuentes_k)
    candidatos_podados = []
    for cand in candidatos:
        subconjuntos = combinations(cand, k)
        if all(frozenset(sub) in frecuentes_set for sub in subconjuntos):
            candidatos_podados.append(cand)

    return candidatos_podados


def generar_itemsets_frecuentes(
    transacciones: List[Set[str]],
    soporte_minimo: float,
) -> Dict[int, Dict[FrozenSet[str], int]]:
    """Ejecuta el algoritmo Apriori completo (Fase 0 + Fase 1) y devuelve
    un diccionario:
        { tamaño_itemset: {itemset: cobertura_absoluta, ...}, ... }

    El filtro en cada nivel se hace por COBERTURA (conteo absoluto de
    transacciones), igual que en el documento de referencia: primero se
    traduce soporte_minimo a una cobertura_minima (Fase 0), y luego se
    exige cobertura(itemset) >= cobertura_minima en cada iteración.
    """
    n = len(transacciones)
    cobertura_minima = calcular_cobertura_minima(n, soporte_minimo)

    itemsets_por_nivel: Dict[int, Dict[FrozenSet[str], int]] = {}

    k = 1
    candidatos = _generar_candidatos_iniciales(transacciones)

    while candidatos:
        frecuentes_k: Dict[FrozenSet[str], int] = {}
        for candidato in candidatos:
            cobertura = _contar_cobertura(transacciones, candidato)
            if cobertura >= cobertura_minima:
                frecuentes_k[candidato] = cobertura

        if not frecuentes_k:
            break

        itemsets_por_nivel[k] = frecuentes_k
        candidatos = _generar_candidatos_siguientes(list(frecuentes_k.keys()), k)
        k += 1

    return itemsets_por_nivel


# ---------------------------------------------------------------------------
# Fase 2: generación de reglas de asociación a partir de itemsets frecuentes
# ---------------------------------------------------------------------------

def _todos_los_itemsets_planos(
    itemsets_por_nivel: Dict[int, Dict[FrozenSet[str], int]]
) -> Dict[FrozenSet[str], int]:
    """Aplana el diccionario por niveles en un solo diccionario itemset->cobertura."""
    plano: Dict[FrozenSet[str], int] = {}
    for nivel in itemsets_por_nivel.values():
        plano.update(nivel)
    return plano


def _subconjuntos_propios_no_vacios(itemset: FrozenSet[str]) -> List[FrozenSet[str]]:
    """Devuelve todos los subconjuntos propios no vacíos de un itemset
    (candidatos a ser el antecedente de una regla). Para un itemset de 3
    elementos esto genera los 3 antecedentes "dobles" (r=2) y los 3
    antecedentes "únicos" (r=1): la 'explosión combinatoria' de 6 reglas
    descrita en el documento de referencia."""
    items = list(itemset)
    subconjuntos = []
    for r in range(1, len(items)):
        for combo in combinations(items, r):
            subconjuntos.append(frozenset(combo))
    return subconjuntos


def generar_reglas(
    itemsets_por_nivel: Dict[int, Dict[FrozenSet[str], int]],
    confianza_minima: float,
    n_transacciones: int,
) -> List[ReglaAsociacion]:
    """Genera todas las reglas A -> B posibles a partir de itemsets
    frecuentes de tamaño >= 2 (Fase 2: "Anatomía de la Confianza" +
    "Explosión Combinatoria"), filtrando por confianza mínima y
    calculando métricas adicionales (soporte, lift, leverage, conviction).

        confianza(A -> B) = cobertura(A U B) / cobertura(A)
    """

    cobertura_de = _todos_los_itemsets_planos(itemsets_por_nivel)
    reglas: List[ReglaAsociacion] = []

    for nivel, itemsets in itemsets_por_nivel.items():
        if nivel < 2:
            continue  # se necesitan al menos 2 items para formar A -> B

        for itemset, cobertura_itemset in itemsets.items():
            for antecedente in _subconjuntos_propios_no_vacios(itemset):
                consecuente = itemset - antecedente

                cobertura_antecedente = cobertura_de.get(antecedente)
                cobertura_consecuente = cobertura_de.get(consecuente)
                if cobertura_antecedente is None or cobertura_consecuente is None:
                    continue
                if cobertura_antecedente == 0:
                    continue

                confianza = cobertura_itemset / cobertura_antecedente
                if confianza < confianza_minima:
                    continue

                soporte_itemset = cobertura_itemset / n_transacciones
                soporte_antecedente = cobertura_antecedente / n_transacciones
                soporte_consecuente = cobertura_consecuente / n_transacciones
                lift = confianza / soporte_consecuente if soporte_consecuente > 0 else float("inf")

                reglas.append(
                    ReglaAsociacion(
                        antecedente=antecedente,
                        consecuente=consecuente,
                        cobertura=cobertura_itemset,
                        cobertura_antecedente=cobertura_antecedente,
                        soporte=soporte_itemset,
                        confianza=confianza,
                        lift=lift,
                    )
                )

    return reglas


def ordenar_mejores_reglas(
    reglas: List[ReglaAsociacion],
    criterio: str = "lift",
    top_n: int = 10,
) -> List[ReglaAsociacion]:
    """Ordena las reglas de mayor a menor según el criterio indicado
    ('lift', 'confianza', 'soporte') y devuelve las top_n mejores."""
    clave = {
        "lift": lambda r: r.lift,
        "confianza": lambda r: r.confianza,
        "soporte": lambda r: r.soporte,
    }.get(criterio, lambda r: r.lift)

    return sorted(reglas, key=clave, reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Función de conveniencia: ejecutar todo el pipeline de una vez
# ---------------------------------------------------------------------------

def ejecutar_apriori(
    transacciones: List[Set[str]],
    soporte_minimo: float = 0.67,
    confianza_minima: float = 0.80,
) -> Tuple[Dict[int, Dict[FrozenSet[str], int]], List[ReglaAsociacion]]:
    """Ejecuta el pipeline completo: Fase 0 (cobertura mínima) + Fase 1
    (itemsets frecuentes) + Fase 2 (reglas). Devuelve
    (itemsets_por_nivel, lista_de_reglas)."""
    n = len(transacciones)
    itemsets_por_nivel = generar_itemsets_frecuentes(transacciones, soporte_minimo)
    reglas = generar_reglas(itemsets_por_nivel, confianza_minima, n_transacciones=n)
    return itemsets_por_nivel, reglas
