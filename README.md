# Proyecto: Reglas de Asociación (Apriori) y Reglas de Clasificación (PRISM)

Este proyecto contiene dos algoritmos de minería de reglas implementados
**desde cero en Python** 
```
proyecto_reglas/
│
├── data/
│   ├── dataset_apriori_supermercado.csv
│   └── dataset_prism_proyectos_software.csv
│
├── apriori/                        # <-- Reglas de asociación
│   ├── datos.py                    # carga + conversión a pares Atributo=Valor
│   ├── algoritmo_apriori.py        # Fase 0, Fase 1 y Fase 2 de Apriori
│   ├── graficos.py
│   ├── main_apriori.py             # script principal (ejecutar este)
│   ├── graficas/
│   └── reglas_apriori_resultado.csv
│
├── prism/                          # <-- Reglas de clasificación
│   ├── datos.py
│   ├── algoritmo_prism.py          # Recubrimiento_secuencial / AprenderUnaRegla / mejorRestriccion
│   ├── graficos.py
│   ├── main_prism.py               # script principal (ejecutar este)
│   ├── graficas/
│   └── reglas_prism_resultado.csv
│
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## 1. Apriori (reglas de asociación — canasta de supermercado)

```bash
cd apriori
python main_apriori.py
```

### Cómo se implementó


1. **Fase 0 — Cobertura mínima:** se traduce el soporte estadístico
   (ej. `0.67`) a un umbral absoluto de registros:
   `cobertura_minima = round(n_transacciones * soporte_minimo)`.
2. **Fase 1 — Itemsets frecuentes:** se generan ítems de tamaño 1
   (`Atributo=Valor`), se filtran por `cobertura >= cobertura_minima`, y
   se combinan iterativamente (join + poda, propiedad Apriori) para
   generar itemsets de tamaño 2, 3, ... hasta que no se generen más.
3. **Fase 2 — Generación de reglas:** para cada itemset frecuente de
   tamaño ≥ 2, se generan TODAS las reglas antecedente→consecuente
   posibles (antecedente único, antecedente doble, etc.), calculando:
   - **Confianza** = cobertura(A∪B) / cobertura(A)
   - **Soporte** = cobertura(A∪B) / n_transacciones
   - **Lift**, **Leverage**, **Conviction**
   Se conservan sólo las reglas con `confianza >= confianza_minima`.

### Parámetros por defecto (`apriori/main_apriori.py`)

Tomados directamente del ejemplo del PDF ("Parámetros del Sistema"):

```python
SOPORTE_MINIMO = 0.67
CONFIANZA_MINIMA = 0.80
```

Con el dataset de 1000 transacciones, esto da `cobertura_minima = 670`.
Puedes cambiar estos valores libremente en la sección `CONFIGURACIÓN` del
script.

## 2. PRISM (reglas de clasificación — éxito de proyectos de software)

```bash
cd prism
python main_prism.py
```

### Cómo se implementó (igual que el pseudocódigo del PDF)

`algoritmo_prism.py` implementa literalmente los tres procedimientos del
documento, con alias de nombres para que el mapeo teoría↔código sea
directo:

```
Recubrimiento_secuencial(Clases, atributos, ejemplos)   -> ejecutar_prism()
AprenderUnaRegla(Clase, Ejemplos, Atributos)             -> _construir_una_regla()
mejorRestriccion(Restricciones, regla)                   -> mejorRestriccion()
```

Lógica exacta:
1. Para cada clase (`Si` / `No`), mientras `E` contenga ejemplos de esa
   clase:
   - Se construye una regla empezando con antecedente vacío.
   - **Mientras la regla cubra algún ejemplo negativo** (precisión < 1.0)
     y todavía haya atributos disponibles: se prueban todas las
     condiciones `Atributo=Valor` posibles, se elige la de **mayor
     confianza** (y en caso de empate, la de **mayor cobertura**), y se
     añade al antecedente.
   - Se guarda la regla y se eliminan de `E` los ejemplos de la clase
     objetivo que la regla cubrió correctamente.
2. Esto genera reglas **100% puras** (`precisión = 1.0`) — el criterio
   clásico de PRISM (Cendrowska, 1987).

### Parámetros por defecto (`prism/main_prism.py`)

```python
PRECISION_OBJETIVO_CONSTRUCCION = 1.0   # regla pura clásica, según el pseudocódigo
COBERTURA_MINIMA = 0.70                 # filtro final de calidad ("soporte")
PRECISION_MINIMA = 0.85                 # filtro final de calidad ("confianza")
```

> **Nota importante:** el algoritmo primero construye reglas puras
> (`precisión = 1.0`) y LUEGO aplica un filtro final de calidad
> (`cobertura_minima`, `precision_minima`). Con este dataset (4 atributos
> categóricos), una regla de 1-2 condiciones cubre típicamente 10-12% de
> los proyectos, por lo que exigir `cobertura_minima = 0.70` descarta
> todas las reglas individuales. El script lo detecta automáticamente y,
> en ese caso, muestra igualmente las mejores reglas que sí cumplen
> `precision_minima = 0.85` (todas con precisión perfecta = 1.0),
> dejando además el archivo `reglas_prism_resultado.csv` con **todas**
> las reglas generadas (marcando cuáles pasan o no el filtro), para que
> puedas ajustar los umbrales según lo que pida tu rúbrica.

## Resultados obtenidos con los parámetros por defecto

**Apriori:** 10 itemsets de tamaño 1 pasan la cobertura mínima (670/1000),
6 de tamaño 2, y se generan 11 reglas con confianza ≥ 0.80 (ej. `Leche=1 →
Pan=1`, confianza 0.94, lift 1.10).

**PRISM:** se generan 60 reglas en total (algoritmo clásico, puras);
4 de ellas tienen precisión = 1.0 y cobertura entre 10% y 12%, por ejemplo:
`SI Metodologia=Agil Y Experiencia_Lider=Alta ENTONCES Exito_Proyecto=Si`
(117/117 instancias, 100% de precisión).

## Resumen conceptual (de las diapositivas "Aprendizaje basado en reglas")

| | **Reglas de asociación (Apriori)** | **Reglas de clasificación (PRISM)** |
|---|---|---|
| Consecuente | Cualquier combinación de pares atributo-valor | Siempre la clase objetivo |
| Objetivo | Descubrir combinaciones frecuentes | Predecir/clasificar instancias futuras |
| Medida guía | Soporte + Confianza (+ Lift) | Confianza (precisión), con cobertura para desempatar |
| Estrategia | Itemsets frecuentes (bottom-up, por niveles) | Recubrimiento secuencial (una regla a la vez, por clase) |
