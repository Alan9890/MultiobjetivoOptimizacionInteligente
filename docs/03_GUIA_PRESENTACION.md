# Guía Completa para la Presentación (10 Diapositivas + Notas de Orador)

Esta guía está diseñada para estructurar la exposición en clase ante el profesor y los compañeros, cuidando rigurosamente el lenguaje técnico solicitado en la sesión.

---

## Regla de Oro durante la Exposición Oral
- **Qué NUNCA decir:** *"Encontramos la solución óptima"*, *"El algoritmo optimizó y halló el óptimo global"*.
- **Qué SIEMPRE decir:** *"Encontramos una solución que entrega un buen compromiso según las preferencias del tomador de decisiones"*, *"El algoritmo aproximó un conjunto de soluciones con buen compromiso sobre el Frente de Pareto"*.
- **Por qué (explicación dada en clase):** En problemas multiobjetivo y metaheurísticos no tenemos forma analítica de demostrar optimalidad global absoluta ni existe una única solución que gane en todos los objetivos a la vez; la calidad depende de las preferencias del decisor.

---

## Guion Diapositiva por Diapositiva

### Diapositiva 1: Portada y Contexto
- **Contenido visual:**
  - Título: *Comparación de Enfoques de Evaluación Multiobjetivo mediante un Algoritmo Genético*.
  - Subtítulo: *Selección Lexicográfica, Suma Ponderada Normalizada y Dominancia de Pareto (NSGA-II)*.
  - Materia: Optimización Inteligente — MIAAD.
- **Notas de orador:**
  > "Buenas tardes. En esta actividad comparamos tres filosofías distintas para evaluar la calidad de una solución en un problema multiobjetivo con 3 objetivos en conflicto, manteniendo exactamente el mismo Algoritmo Genético base. Nuestro propósito es analizar qué tipo de compromiso entrega cada enfoque según las preferencias del tomador de decisiones."

### Diapositiva 2: El Problema de Inspección en Línea SMT (3 Objetivos en Conflicto)
- **Contenido visual:**
  - Vector de decisión: $\mathbf{x} = (s_1, s_2, s_3, s_4, v) \in [0, 1]^5$ (4 sensibilidades ópticas por clase de defecto y la velocidad de línea $v$).
  - Ecuaciones de los 3 objetivos a minimizar:
    - $f_1(\mathbf{x}) = (1 + v)\sum a_i(1-s_i)^2 \in [0, 5.6]$ — *Defectos que se escapan al cliente*.
    - $f_2(\mathbf{x}) = \sum b_i s_i^2 \in [0, 3.2]$ — *Falsas alarmas (retrabajos innecesarios)*.
    - $f_3(\mathbf{x}) = (1 - v) + 0.3\bar{s} \in [0, 1.3]$ — *Tiempo y costo de ciclo de inspección*.
- **Notas de orador:**
  > "Si subimos la sensibilidad $s_i \to 1$ y bajamos la velocidad $v \to 0$, eliminamos los defectos escapados ($f_1 \to 0$), pero disparamos las falsas alarmas ($f_2 \to 3.2$) y alentamos toda la línea ($f_3 \to 1.3$). Si por el contrario aceleramos la línea ($v \to 1$) y apagamos sensores ($s_i \to 0$), el tiempo y las falsas alarmas bajan a cero, pero se escapan todos los defectos ($f_1 \to 5.6$). Existe un conflicto real e inevitable."

### Diapositiva 3: Los 3 Enfoques de Calidad bajo la Misma Metaheurística
- **Contenido visual:**
  - Tabla comparativa de los 3 mecanismos:
    1. **Selección Lexicográfica:** Prioridad absoluta ($F_1 > F_2 > F_3$ y permutación $F_3 > F_2 > F_1$) con tolerancia relativa de empate $\tau = 10^{-3}$.
    2. **Suma Ponderada:** Previa normalización min-max a $[0, 1]$, $F(\mathbf{x}) = \sum w_i \hat{f}_i(\mathbf{x})$ con 3 vectores de pesos: equilibrado $(1/3, 1/3, 1/3)$, sesgo a $F_1$ $(0.70, 0.15, 0.15)$ y sesgo a $F_3$ $(0.15, 0.15, 0.70)$.
    3. **Dominancia de Pareto (NSGA-II):** Ordenamiento no dominado por frentes (Frente 1, 2, ...), *crowding distance* para diversidad y truncamiento elitista $R_t = P_t \cup Q_t$.
- **Notas de orador:**
  > "Es importante recalcar que no cambiamos de metaheurística: la población inicial por semilla, la cruza BLX-0.5 y la mutación gaussiana son idénticas en los 6 experimentos. Lo único que cambia es cómo el algoritmo decide qué individuo es preferible a otro."

### Diapositiva 4: Configuración Experimental Controlada
- **Contenido visual:**
  - Parámetros fijos: Población $N=100$, Generaciones $G=100$, Cruza BLX-$\alpha$ ($\alpha=0.5, p_c=0.90$), Mutación Gaussiana ($\sigma=0.1, p_m=1/5$).
  - Rigor estadístico: **10 semillas fijas reportadas** (`101` a `1010`) $\times$ 6 configuraciones = **60 corridas independientes**.
  - Métrica de cobertura común: Hipervolumen 3D con punto de referencia fijo $\mathbf{r} = [6.0, 3.5, 1.5]$.
- **Notas de orador:**
  > "Para una comparación pareada justa, en cada semilla las 6 configuraciones arrancan exactamente con los mismos 100 individuos iniciales."

### Diapositiva 5: Exploración del Espacio de Objetivos 3D y 2D
- **Contenido visual:** Insertar `results/fig1_espacio_objetivos_3d_2d.png`.
- **Notas de orador:**
  > "En esta primera gráfica observamos el espacio tridimensional y sus tres proyecciones 2D con el mismo rango de ejes. Noten cómo las soluciones de la Selección Lexicográfica (triángulos naranjas y magentas) se van a las esquinas extremas del espacio; las tres Sumas Ponderadas (cuadrado, rombo y cruz) convergen cada una a un punto aislado distinto según el peso asignado; mientras que NSGA-II despliega una nube que cubre toda la superficie de compromiso."

### Diapositiva 6: Frente de Pareto de NSGA-II y Regiones de Interés (ROI)
- **Contenido visual:** Insertar `results/fig2_frente_pareto_nsga2.png`.
- **Notas de orador:**
  > "Aquí analizamos exclusivamente a NSGA-II. En cruces grises vemos soluciones dominadas iniciales (Frentes secundarios), y en colores el Frente 1 no dominado. Como vimos en clase, entregarle 100 soluciones al usuario abre el concepto de **Región de Interés (ROI)**: señalamos la ROI A para un tomador de decisiones que exige cero defectos, la ROI B de compromiso central equilibrado, y la ROI C para alta velocidad de producción."

### Diapositiva 7: Comparación Cuantitativa por Objetivo e Hipervolumen
- **Contenido visual:** Insertar `results/fig3_barras_objetivos.png` y tabla resumen de medias $\pm$ desviación estándar.
- **Notas de orador:**
  > "En las barras con barras de error sobre las 10 semillas se aprecia claramente el sacrificio entre objetivos. Además, en el panel D vemos el resultado más contundente: NSGA-II alcanza un Hipervolumen de $26.101$ con $99.9$ soluciones no dominadas por corrida, frente a $17.687$ de la suma ponderada equilibrada y apenas $0.824$ de la selección lexicográfica $F_1 > F_2 > F_3$."

### Diapositiva 8: Convergencia por Generación e Indicador de Sesgo
- **Contenido visual:** Insertar `results/fig4_convergencia_generaciones.png` y `results/fig5_indicador_sesgo_tradeoff.png`.
- **Notas de orador:**
  > "En la gráfica de convergencia ocurre un fenómeno muy revelador: en el panel derecho todas las estrategias convergen limpiamente antes de la generación 30 en su propia métrica escalar. Pero en el panel izquierdo vemos que el Hipervolumen de la población *sube* en NSGA-II hasta $26.10$, mientras que *baja* en la Suma Ponderada y Lexicográfica. ¿Por qué baja el hipervolumen poblacional en esos métodos? Porque al inicio $P_0$ tenía 100 puntos dispersos, y conforme avanzan las generaciones, la suma ponderada y la lexicográfica colapsan a los 100 individuos en un único punto, perdiendo la cobertura del resto del frente."

### Diapositiva 9: Ventajas y Deficiencias de cada Enfoque
- **Contenido visual:** Cuadro comparativo de los 3 párrafos síntesis (Sección 6 del `README.md`).
- **Notas de orador:**
  > "En síntesis: la Selección Lexicográfica es ideal solo cuando hay una prioridad absoluta innegociable, pero destruye los objetivos secundarios (sacrificios de hasta $-688\%$). La Suma Ponderada es rápida y considera todos los objetivos, pero exige conocer los pesos de antemano, colapsa la población a una sola solución y sesga la búsqueda hacia el mayor peso. NSGA-II conserva la diversidad de compromisos ($\approx 100$ soluciones no dominadas), permitiendo elegir la Región de Interés a posteriori a cambio de mayor cómputo por generación."

### Diapositiva 10: Conclusiones y Conexión con Lógica Difusa Compensatoria
- **Contenido visual:**
  - Conclusión en términos de *"solución que entrega un buen compromiso según las preferencias del tomador de decisiones"*.
  - Perspectiva hacia el siguiente tema: **Lógica Difusa Compensatoria** (uso de la media geométrica para lograr equilibrio entre objetivos sin el sesgo agresivo de la suma ponderada aditiva).
- **Notas de orador:**
  > "Concluimos que ningún método entrega una 'solución óptima' universal, sino soluciones que entregan un buen compromiso según cómo se modelen las preferencias del decisor. Finalmente, estos resultados motivan el uso de la Lógica Difusa Compensatoria que veremos a continuación en el curso, la cual busca combinar la rapidez de una función agregada con un balance geométrico que evite el sesgo extremo observado en la suma ponderada."
