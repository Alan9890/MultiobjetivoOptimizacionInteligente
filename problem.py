"""
problem.py
Definición aislada del problema de optimización multiobjetivo (3 objetivos a minimizar).
Por defecto: Configuración de inspección en una línea SMT (5 variables continuas en [0, 1]).
Incluye además DTLZ2 (3 objetivos) como problema alternativo de validación.

Nota terminológica: En este proyecto nunca hablamos de "solución óptima", sino de
"solución que entrega un buen compromiso según las preferencias del tomador de decisiones".
"""

from __future__ import annotations
import numpy as np

# ==============================================================================
# 1. CONFIGURACIÓN DEL PROBLEMA ACTIVO ("SMT" o "DTLZ2")
# ==============================================================================
ACTIVE_PROBLEM = "SMT"

# Coeficientes del problema de inspección SMT
A_COEF = np.array([1.0, 0.8, 0.6, 0.4], dtype=float)
B_COEF = np.array([0.5, 0.7, 0.9, 1.1], dtype=float)

if ACTIVE_PROBLEM == "SMT":
    PROBLEM_NAME = "Configuración de Inspección en Línea SMT"
    N_VARS = 5
    VAR_NAMES = ["s1", "s2", "s3", "s4", "v"]
    N_OBJ = 3
    OBJ_NAMES = [
        "f1: Defectos escapados [índice adim.]",
        "f2: Falsas alarmas [índice adim.]",
        "f3: Tiempo/costo de inspección [unid. norm.]",
    ]
    OBJ_SHORT_NAMES = ["f1", "f2", "f3"]
    BOUNDS: list[tuple[float, float]] = [(0.0, 1.0)] * N_VARS

    # Límites analíticos exactos de cada objetivo en el dominio [0, 1]^5
    # f1 in [0, (1+1)*(1.0+0.8+0.6+0.4) = 5.6]
    # f2 in [0, 0.5+0.7+0.9+1.1 = 3.2]
    # f3 in [0, 1.0 + 0.3*1.0 = 1.3]
    OBJ_MIN = np.array([0.0, 0.0, 0.0], dtype=float)
    OBJ_MAX = np.array([5.6, 3.2, 1.3], dtype=float)

    # Punto de referencia común para el cálculo de hipervolumen:
    # En espacio original (domina estrictamente al punto nadir [5.6, 3.2, 1.3])
    HV_REF_POINT = np.array([6.0, 3.5, 1.5], dtype=float)
    # En espacio normalizado [0, 1]^3 (con holgura del 10% sobre el máximo 1.0)
    HV_REF_POINT_NORM = np.array([1.1, 1.1, 1.1], dtype=float)

else:
    PROBLEM_NAME = "Benchmark DTLZ2 (3 objetivos)"
    N_VARS = 12
    VAR_NAMES = [f"x{i+1}" for i in range(N_VARS)]
    N_OBJ = 3
    OBJ_NAMES = [
        "f1: Objetivo DTLZ2-1 [adim.]",
        "f2: Objetivo DTLZ2-2 [adim.]",
        "f3: Objetivo DTLZ2-3 [adim.]",
    ]
    OBJ_SHORT_NAMES = ["f1", "f2", "f3"]
    BOUNDS = [(0.0, 1.0)] * N_VARS
    OBJ_MIN = np.array([0.0, 0.0, 0.0], dtype=float)
    OBJ_MAX = np.array([3.5, 3.5, 3.5], dtype=float)
    HV_REF_POINT = np.array([4.0, 4.0, 4.0], dtype=float)
    HV_REF_POINT_NORM = np.array([1.1, 1.1, 1.1], dtype=float)


# ==============================================================================
# 2. FUNCIONES DE EVALUACIÓN
# ==============================================================================
def evaluate_smt(x: list[float] | np.ndarray) -> tuple[float, float, float]:
    """
    Evalúa una configuración x = (s1, s2, s3, s4, v) en [0, 1]^5 para la línea SMT.
    Todos los objetivos se minimizan:
      - f1: defectos que se escapan = (1 + v) * sum_i a_i * (1 - s_i)^2
      - f2: falsas alarmas          = sum_i b_i * s_i^2
      - f3: tiempo/costo inspección = (1 - v) + 0.3 * mean(s)
    """
    s0, s1, s2, s3, v = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
    d0, d1, d2, d3 = 1.0 - s0, 1.0 - s1, 1.0 - s2, 1.0 - s3

    f1 = (1.0 + v) * (1.0 * d0 * d0 + 0.8 * d1 * d1 + 0.6 * d2 * d2 + 0.4 * d3 * d3)
    f2 = 0.5 * s0 * s0 + 0.7 * s1 * s1 + 0.9 * s2 * s2 + 1.1 * s3 * s3
    f3 = (1.0 - v) + 0.075 * (s0 + s1 + s2 + s3)
    return (f1, f2, f3)


def evaluate_dtlz2(x: list[float] | np.ndarray) -> tuple[float, float, float]:
    """
    Problema alternativo de validación: DTLZ2 con M = 3 objetivos a minimizar.
    El Frente de Pareto verdadero satisface f1^2 + f2^2 + f3^2 = 1 (con x_m = 0.5).
    """
    arr = np.asarray(x, dtype=float)
    xm = arr[2:]
    g = float(np.sum((xm - 0.5) ** 2))

    f1 = (1.0 + g) * np.cos(arr[0] * np.pi / 2.0) * np.cos(arr[1] * np.pi / 2.0)
    f2 = (1.0 + g) * np.cos(arr[0] * np.pi / 2.0) * np.sin(arr[1] * np.pi / 2.0)
    f3 = (1.0 + g) * np.sin(arr[0] * np.pi / 2.0)
    return (float(f1), float(f2), float(f3))


def evaluate(x: list[float] | np.ndarray) -> tuple[float, float, float]:
    """Función principal de evaluación según el problema activo."""
    if ACTIVE_PROBLEM == "SMT":
        return evaluate_smt(x)
    return evaluate_dtlz2(x)


def normalize_objectives(f_vec: list[float] | np.ndarray) -> np.ndarray:
    """Normalización min-max de un vector (o matriz) de objetivos al rango [0, 1]."""
    arr = np.asarray(f_vec, dtype=float)
    denom = np.where((OBJ_MAX - OBJ_MIN) > 1e-12, OBJ_MAX - OBJ_MIN, 1.0)
    return (arr - OBJ_MIN) / denom


if __name__ == "__main__":
    print(f"=== Prueba rápida de problem.py ({PROBLEM_NAME}) ===")
    casos = {
        "Sensibilidad máxima, baja velocidad (s=1, v=0)": [1.0, 1.0, 1.0, 1.0, 0.0],
        "Sensibilidad mínima, alta velocidad (s=0, v=1)": [0.0, 0.0, 0.0, 0.0, 1.0],
        "Punto intermedio equilibrado (s=0.5, v=0.5)":    [0.5, 0.5, 0.5, 0.5, 0.5],
    }
    for nombre, x_test in casos.items():
        vals = evaluate(x_test)
        vals_norm = normalize_objectives(vals)
        print(f"- {nombre}:")
        print(f"    x      = {x_test}")
        print(f"    f(x)   = ({vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f})")
        print(f"    f_norm = ({vals_norm[0]:.4f}, {vals_norm[1]:.4f}, {vals_norm[2]:.4f})")
