# Proyecto: comparación de enfoques multiobjetivo con un algoritmo genético

> Uso: pega este documento completo como primer mensaje en Claude (o Claude Code) y pide que lo ejecute paso a paso. Cuando llegue el problema oficial del profesor, ve a la sección 9.

## 1. Rol y contexto

Eres un asistente experto en optimización metaheurística y Python. Estoy en una maestría en IA y Analítica de Datos (materia: Optimización Inteligente). Debo entregar y presentar una actividad que compara tres formas de evaluar la calidad de una solución en un problema **multiobjetivo con al menos 3 objetivos**, usando **la misma metaheurística** (algoritmo genético) en los tres casos.

Terminología obligatoria en textos y gráficas: nunca decir "solución óptima"; decir "solución que entrega un buen compromiso según las preferencias del tomador de decisiones".

## 2. Los tres enfoques a comparar

Todos minimizan. Solo cambia la forma de calcular la calidad (fitness/selección); cruza, mutación, población y generaciones son idénticas.

1. **Selección lexicográfica.** Orden de prioridad F1 > F2 > F3. Se compara primero F1; solo si hay empate (con tolerancia configurable, p. ej. 1e-3 relativo) se pasa a F2, y luego a F3.
2. **Suma ponderada.** `F = w1*f1 + w2*f2 + w3*f3` con `sum(w)=1`. Normalizar cada objetivo (min-max con límites estimados) antes de sumar. Probar al menos 3 configuraciones de pesos: parejos (1/3 cada uno), sesgado a f1 y sesgado a f3.
3. **Dominancia de Pareto con NSGA-II.** Ordenamiento no dominado (Frente 1, 2, ...), crowding distance para desempatar y truncamiento elitista (`R_t = P_t ∪ Q_t`, quedarse con los N mejores). Usar `deap.tools.selNSGA2`.

## 3. Problema por defecto (si aún no tengo el oficial)

Configuración de inspección en una línea SMT, 5 variables continuas en [0,1]:
`x = (s1, s2, s3, s4, v)`: sensibilidad de detección para 4 clases de defecto y velocidad de inspección.

Objetivos (3, todos a minimizar; el asistente puede ajustar coeficientes, pero deben existir conflictos reales):

- `f1` (defectos que se escapan) = `(1 + v) * sum_i a_i * (1 - s_i)^2`, con `a = [1.0, 0.8, 0.6, 0.4]`
- `f2` (falsas alarmas) = `sum_i b_i * s_i^2`, con `b = [0.5, 0.7, 0.9, 1.1]`
- `f3` (tiempo/costo de inspección) = `(1 - v) + 0.3 * mean(s)`

Alternativa si algo falla: DTLZ2 con 3 objetivos (frente conocido, sirve de validación).

Dejar el problema aislado en un módulo `problem.py` con una función `evaluate(x) -> (f1, f2, f3)` y los límites de cada variable, para poder cambiarlo sin tocar el resto.

## 4. Estructura del repositorio

```
proyecto/
  problem.py          # evaluate(x), bounds, n_obj
  ga_core.py          # cruza, mutación, inicialización, bucle genético genérico
  strategies.py       # lexicográfica, suma ponderada, NSGA-II
  run_experiments.py  # corre todo con múltiples semillas, guarda CSV
  analysis.py         # métricas y gráficas
  results/            # CSV y PNG
  README.md
```

## 5. Parámetros del experimento

- Población 100, 100 generaciones (configurable en un solo diccionario).
- Cruza: blend (BLX-α, α=0.5) o SBX; mutación gaussiana (σ=0.1) con probabilidad 1/n_vars; respetar límites.
- 10 semillas por configuración (semillas fijas y reportadas).
- Configuraciones: lexicográfica (1 orden de prioridad, más una segunda permutación para mostrar sensibilidad), ponderada (3 conjuntos de pesos), NSGA-II.

## 6. Métricas y comparación

Por cada corrida guardar los vectores de objetivos de la población final (o del archivo de no dominadas) y calcular:

- Valor medio de f1, f2, f3 de la mejor solución de cada método (media y desviación sobre semillas).
- Número de soluciones no dominadas encontradas.
- Hipervolumen del conjunto no dominado final (punto de referencia común, documentado), idéntico para todos los métodos.
- Un indicador de sesgo: cuánto mejora cada objetivo respecto a NSGA-II o a la solución "equilibrada".

## 7. Gráficas requeridas

1. Dispersión 3D (y 3 proyecciones 2D) del espacio de objetivos, mismo rango de ejes, un color por método.
2. Frente de Pareto de NSGA-II, resaltando Frente 1.
3. Barras por objetivo comparando métodos (media ± desviación).
4. Convergencia por generación (hipervolumen o mejor valor escalarizado).

Estilo: títulos en español, ejes con nombre y unidades, leyendas legibles, guardar en PNG a 200 dpi.

## 8. Resultados esperados (hipótesis a contrastar, no a forzar)

- Lexicográfica: mejora mucho el objetivo prioritario y descuida los demás.
- Suma ponderada: se sesga hacia el objetivo con mayor peso; los pesos hay que definirlos de antemano.
- NSGA-II: entrega un conjunto de compromisos; el tomador de decisiones elige después (región de interés).

Si los datos contradicen una hipótesis, reportarlo tal cual y explicar posibles causas.

## 9. Cómo cambiar al problema oficial

Reemplazar `problem.py` con la instancia y la función de evaluación del profesor, ajustar `bounds`, `n_obj` y los límites de normalización. No tocar `ga_core.py` ni `strategies.py`. Re-ejecutar `run_experiments.py` y `analysis.py`. Si el problema oficial usa variables discretas o permutaciones, adaptar solo cruza y mutación en `ga_core.py` y mantenerlas iguales en los tres métodos.

## 10. Entregables

1. Código ejecutable (Python 3.10+, `deap`, `numpy`, `pandas`, `matplotlib`), con `requirements.txt` y semillas fijas.
2. Carpeta `results/` con CSV y gráficas.
3. Esquema de presentación (8 a 10 diapositivas): problema y objetivos, los 3 enfoques, configuración experimental, resultados por método, comparación, ventajas y deficiencias, conclusión en términos de "buen compromiso", cierre.
4. Un párrafo corto por método con sus ventajas y deficiencias, escrito con los datos obtenidos.

## 11. Forma de trabajo pedida al asistente

- Trabaja por etapas: primero `problem.py` y `ga_core.py` con una prueba rápida, luego `strategies.py`, luego experimentos, luego análisis.
- Después de cada etapa, muestra cómo ejecutarla y qué salida esperar.
- Comenta el código en español, claro y breve.
- No inventes resultados: reporta solo lo que produzca la ejecución real.
