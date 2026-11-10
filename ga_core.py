"""
ga_core.py
Núcleo del Algoritmo Genético (AG): inicialización, operadores de cruza y mutación
con respeto de límites, funciones auxiliares de dominancia de Pareto, hipervolumen
y bucle evolutivo genérico.

Todos los enfoques (Lexicográfico, Suma Ponderada y NSGA-II) comparten exactamente
este mismo núcleo, cambiando únicamente la estrategia de selección/calidad.
"""

from __future__ import annotations
import copy
import random
from typing import Any, Protocol
import numpy as np
from deap import base, creator, tools

try:
    import moocore
    _HAS_MOOCORE = True
except ImportError:
    _HAS_MOOCORE = False

import problem

# ==============================================================================
# 1. DICCIONARIO ÚNICO DE CONFIGURACIÓN EXPERIMENTAL
# ==============================================================================
GA_CONFIG: dict[str, Any] = {
    "pop_size": 100,                     # Tamaño de la población (N)
    "n_gen": 100,                        # Número de generaciones (G)
    "cx_type": "blx",                    # Tipo de cruza: "blx" (BLX-alpha) o "sbx"
    "blx_alpha": 0.5,                    # Parámetro alpha para cruza blend (BLX-0.5)
    "sbx_eta": 15.0,                     # Parámetro eta para cruza SBX
    "cx_prob": 0.9,                      # Probabilidad de cruza entre pares de padres
    "mut_sigma": 0.1,                    # Desviación estándar de mutación gaussiana
    "mut_indpb": 1.0 / problem.N_VARS,   # Probabilidad de mutar cada gen (1 / n_vars)
    "mut_prob": 1.0,                     # Se evalúa mutación por gen en cada descendiente
    "lex_tol": 1e-3,                     # Tolerancia relativa para empate lexicográfico
    "seeds": [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010],  # 10 semillas fijas
}

# ==============================================================================
# 2. REGISTRO DE CLASES EN DEAP
# ==============================================================================
if not hasattr(creator, "FitnessMinMO"):
    creator.create("FitnessMinMO", base.Fitness, weights=(-1.0,) * problem.N_OBJ)

if not hasattr(creator, "IndividualMO"):
    creator.create("IndividualMO", list, fitness=creator.FitnessMinMO)


# ==============================================================================
# 3. PROTOCOLO DE ESTRATEGIA DE SELECCIÓN/CALIDAD
# ==============================================================================
class SelectionStrategy(Protocol):
    name: str
    label: str

    def prepare_population(self, pop: list[Any]) -> list[Any]:
        """Asigna atributos de calidad/ranking a la población actual."""
        ...

    def select_parents(self, pop: list[Any], k: int, rng: random.Random) -> list[Any]:
        """Selecciona k padres mediante torneo usando el criterio del método."""
        ...

    def select_survivors(self, combined_pop: list[Any], k: int) -> list[Any]:
        """Truncamiento elitista R_t = P_t U Q_t para conservar los k mejores."""
        ...

    def get_representative_solution(self, pop: list[Any]) -> Any:
        """Devuelve la solución representativa que entrega un buen compromiso para este método."""
        ...


# ==============================================================================
# 4. OPERADORES GENÉTICOS E INICIALIZACIÓN
# ==============================================================================
def clip_individual(ind: list[float], bounds: list[tuple[float, float]]) -> None:
    """Acota cada variable de decisión del individuo dentro de sus límites [low, high]."""
    for i, (low, high) in enumerate(bounds):
        if ind[i] < low:
            ind[i] = low
        elif ind[i] > high:
            ind[i] = high


def create_individual(rng: random.Random, bounds: list[tuple[float, float]]) -> Any:
    """Crea un individuo aleatorio uniforme dentro de los límites del problema."""
    genes = [rng.uniform(low, high) for (low, high) in bounds]
    return creator.IndividualMO(genes)


def init_population(pop_size: int, rng: random.Random, bounds: list[tuple[float, float]]) -> list[Any]:
    """Genera la población inicial P_0 de tamaño pop_size."""
    return [create_individual(rng, bounds) for _ in range(pop_size)]


def evaluate_population(pop: list[Any]) -> None:
    """Evalúa todos los individuos cuyo fitness no sea válido aún."""
    for ind in pop:
        if not ind.fitness.valid:
            ind.fitness.values = problem.evaluate(ind)


def clone_individual(ind: Any) -> Any:
    """Clona rápidamente un individuo y su fitness sin la sobrecarga de copy.deepcopy."""
    cloned = creator.IndividualMO(ind)
    if ind.fitness.valid:
        cloned.fitness.values = ind.fitness.values
    return cloned


def apply_crossover_and_mutation(
    parents: list[Any],
    config: dict[str, Any],
    bounds: list[tuple[float, float]],
) -> list[Any]:
    """
    Aplica cruza (BLX-alpha o SBX) y mutación gaussiana (sigma=0.1, indpb=1/n_vars)
    respetando estrictamente los límites de cada variable.
    """
    offspring = [clone_individual(ind) for ind in parents]
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]

    # 1. Cruza por pares consecutivos
    for i in range(0, len(offspring) - 1, 2):
        if random.random() < config["cx_prob"]:
            c1, c2 = offspring[i], offspring[i + 1]
            if config.get("cx_type", "blx") == "blx":
                tools.cxBlend(c1, c2, alpha=config["blx_alpha"])
            else:
                tools.cxSimulatedBinaryBounded(c1, c2, eta=config["sbx_eta"], low=lows, up=highs)
            clip_individual(c1, bounds)
            clip_individual(c2, bounds)
            del c1.fitness.values
            del c2.fitness.values

    # 2. Mutación gaussiana por gen con probabilidad 1 / n_vars
    for child in offspring:
        if random.random() < config["mut_prob"]:
            tools.mutGaussian(
                child,
                mu=0.0,
                sigma=config["mut_sigma"],
                indpb=config["mut_indpb"],
            )
            clip_individual(child, bounds)
            del child.fitness.values

    return offspring


def deduplicate_population(pop: list[Any], decimals: int = 5) -> tuple[list[Any], list[Any]]:
    """
    Separa individuos con genotipos distintos (redondeando a `decimals` cifras)
    de clones exactos en R_t = P_t U Q_t. Útil en NSGA-II para evitar distancia
    de hacinamiento nula por copias idénticas padre-hijo.
    """
    seen: set[tuple[float, ...]] = set()
    unique: list[Any] = []
    duplicates: list[Any] = []
    for ind in pop:
        key = tuple(round(g, decimals) for g in ind)
        if key not in seen:
            seen.add(key)
            unique.append(ind)
        else:
            duplicates.append(ind)
    return unique, duplicates


# ==============================================================================
# 5. UTILIDADES MULTIOBJETIVO: DOMINANCIA E HIPERVOLUMEN
# ==============================================================================
def get_nondominated_vectors(obj_matrix: np.ndarray, tol_decimals: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Dado un arreglo (N, M) de vectores de objetivos a minimizar, devuelve:
      - mask: arreglo booleano de longitud N (True si la solución es no dominada)
      - nd_matrix: submatriz con los vectores no dominados distintos (con resolución `tol_decimals`)
    Una solución A domina a B si A <= B en todo objetivo y A < B en al menos uno.
    """
    pts = np.asarray(obj_matrix, dtype=float)
    if _HAS_MOOCORE and len(pts) > 0:
        is_nd = np.asarray(moocore.is_nondominated(pts), dtype=bool)
    else:
        n = pts.shape[0]
        is_nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_nd[i]:
                continue
            le_all = np.all(pts <= pts[i], axis=1)
            lt_any = np.any(pts < pts[i], axis=1)
            if np.any(le_all & lt_any):
                is_nd[i] = False

    nd_pts = pts[is_nd]
    if len(nd_pts) > 1:
        _, unique_idx = np.unique(np.round(nd_pts, decimals=tol_decimals), axis=0, return_index=True)
        nd_pts = nd_pts[np.sort(unique_idx)]
    return is_nd, nd_pts


def compute_hypervolume_3d(nd_points: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Calcula el hipervolumen exacto en 3D (minimización) del conjunto `nd_points`
    respecto al punto de referencia `ref_point`. Usa `moocore` (C nativo) si está
    disponible, o barrido exacto por planos en caso contrario.
    """
    if len(nd_points) == 0:
        return 0.0

    pts = np.asarray(nd_points, dtype=float)
    ref = np.asarray(ref_point, dtype=float)

    valid = np.all(pts < ref, axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    if _HAS_MOOCORE:
        return float(moocore.hypervolume(pts, ref=ref))

    order = np.argsort(pts[:, 2])
    pts = pts[order]

    hv = 0.0
    n = len(pts)
    for i in range(n):
        z_curr = pts[i, 2]
        z_next = pts[i + 1, 2] if (i + 1) < n else ref[2]
        dz = z_next - z_curr
        if dz <= 0.0:
            continue
        area_2d = _compute_hv_2d(pts[: i + 1, :2], ref[:2])
        hv += area_2d * dz

    return float(hv)


def _compute_hv_2d(pts_2d: np.ndarray, ref_2d: np.ndarray) -> float:
    """Calcula el hipervolumen (área) 2D exacto respecto a ref_2d (minimización)."""
    order = np.argsort(pts_2d[:, 0])
    sorted_pts = pts_2d[order]

    area = 0.0
    current_max_y = ref_2d[1]
    for x, y in sorted_pts:
        if y < current_max_y:
            area += (ref_2d[0] - x) * (current_max_y - y)
            current_max_y = y
    return float(area)


# ==============================================================================
# 6. BUCLE PRINCIPAL DEL ALGORITMO GENÉTICO
# ==============================================================================
def run_ga(
    strategy: SelectionStrategy,
    seed: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Ejecuta el Algoritmo Genético con la estrategia dada y una semilla fija.
    Devuelve la población final, la solución representativa y el historial por generación.
    """
    cfg = copy.deepcopy(GA_CONFIG if config is None else config)
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    pop_size = cfg["pop_size"]
    n_gen = cfg["n_gen"]
    bounds = problem.BOUNDS

    # 1. Inicialización y evaluación de P_0
    pop = init_population(pop_size, rng, bounds)
    evaluate_population(pop)
    pop = strategy.prepare_population(pop)

    history: list[dict[str, float]] = []

    def record_generation(gen_idx: int, current_pop: list[Any]) -> None:
        objs = np.array([ind.fitness.values for ind in current_pop], dtype=float)
        objs_norm = problem.normalize_objectives(objs)
        _, nd_raw = get_nondominated_vectors(objs)
        nd_norm = problem.normalize_objectives(nd_raw)

        hv_raw = compute_hypervolume_3d(nd_raw, problem.HV_REF_POINT)
        hv_norm = compute_hypervolume_3d(nd_norm, problem.HV_REF_POINT_NORM)
        equal_scalar = float(np.min(np.mean(objs_norm, axis=1)))
        rep_ind = strategy.get_representative_solution(current_pop)
        rep_f = rep_ind.fitness.values

        history.append(
            {
                "gen": gen_idx,
                "hv_raw": hv_raw,
                "hv_norm": hv_norm,
                "n_nd": len(nd_raw),
                "best_equal_scalar": equal_scalar,
                "rep_f1": rep_f[0],
                "rep_f2": rep_f[1],
                "rep_f3": rep_f[2],
            }
        )

    record_generation(0, pop)

    # 2. Ciclo evolutivo por generaciones
    for gen in range(1, n_gen + 1):
        # Selección de padres
        parents = strategy.select_parents(pop, pop_size, rng)

        # Cruza y mutación (idénticas para todas las estrategias)
        offspring = apply_crossover_and_mutation(parents, cfg, bounds)
        evaluate_population(offspring)

        # Combinación elitista R_t = P_t U Q_t y selección de N sobrevivientes
        combined = pop + offspring
        pop = strategy.select_survivors(combined, pop_size)

        record_generation(gen, pop)

    rep_solution = strategy.get_representative_solution(pop)
    final_objs = np.array([ind.fitness.values for ind in pop], dtype=float)
    is_nd, nd_objs = get_nondominated_vectors(final_objs)

    return {
        "strategy_name": strategy.name,
        "strategy_label": strategy.label,
        "seed": seed,
        "final_pop": pop,
        "final_objs": final_objs,
        "is_nd": is_nd,
        "nd_objs": nd_objs,
        "representative": rep_solution,
        "representative_objs": rep_solution.fitness.values,
        "history": history,
    }


if __name__ == "__main__":
    print("=== Prueba rápida de ga_core.py ===")
    rng_test = random.Random(101)
    p0 = init_population(10, rng_test, problem.BOUNDS)
    evaluate_population(p0)
    objs_0 = np.array([ind.fitness.values for ind in p0])
    mask_nd, pts_nd = get_nondominated_vectors(objs_0)
    hv_0 = compute_hypervolume_3d(pts_nd, problem.HV_REF_POINT)
    print(f"Población de prueba inicializada: {len(p0)} individuos.")
    print(f"Primer individuo x = {[round(g, 3) for g in p0[0]]} -> f(x) = {[round(v, 4) for v in p0[0].fitness.values]}")
    print(f"Soluciones no dominadas en P0: {len(pts_nd)} / {len(p0)}")
    print(f"Hipervolumen 3D inicial (ref={problem.HV_REF_POINT.tolist()}): {hv_0:.4f}")
