"""
strategies.py
Implementación de los tres enfoques para evaluar la calidad y seleccionar soluciones
en el Algoritmo Genético Multiobjetivo:
  1. Selección Lexicográfica (con orden de prioridad y tolerancia relativa configurable).
  2. Suma Ponderada (con normalización min-max previa y pesos que suman 1).
  3. Dominancia de Pareto con NSGA-II (`deap.tools.selNSGA2` y crowding distance).

Nota terminológica: Ningún método busca una "solución óptima" absoluta, sino una
"solución que entrega un buen compromiso según las preferencias del tomador de decisiones".
"""

from __future__ import annotations
import math
import random
from typing import Any
import numpy as np
from deap import tools

import problem
import ga_core


# ==============================================================================
# 1. ENFOQUE 1: SELECCIÓN LEXICOGRÁFICA
# ==============================================================================
class LexicographicStrategy:
    """
    Evalúa y selecciona soluciones comparando los objetivos en un orden estricto
    de prioridad (p. ej. F1 > F2 > F3). Solo si existe empate dentro de una
    tolerancia relativa `tol` (p. ej. 1e-3 del rango del objetivo) se pasa al
    siguiente objetivo en la jerarquía.
    """

    def __init__(
        self,
        priority: tuple[int, int, int] = (0, 1, 2),
        tol: float = 1e-3,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        self.priority = priority
        self.tol = tol
        ord_str = " > ".join(f"F{i+1}" for i in priority)
        self.name = name or f"Lex_{'_'.join(f'F{i+1}' for i in priority)}"
        self.label = label or f"Lexicográfica ({ord_str})"

        self._obj_min = tuple(float(v) for v in problem.OBJ_MIN)
        self._inv_range = tuple(
            1.0 / float(mx - mn) if (mx - mn) > 1e-12 else 1.0
            for mn, mx in zip(problem.OBJ_MIN, problem.OBJ_MAX)
        )

    def _lex_key(self, ind: Any) -> tuple[float, ...]:
        """
        Construye una tupla clave transitiva para el ordenamiento lexicográfico con
        tolerancia relativa `tol`:
          - Los objetivos se normalizan al rango [0, 1] y se discretizan en escalones
            de ancho `tol` (empate si caen dentro del mismo escalón de tolerancia `tol`).
          - Si empatan en el primer objetivo, desempata el segundo; si empatan en el
            segundo, desempata el tercero; y finalmente el valor exacto.
        """
        fv = ind.fitness.values
        fn0 = (fv[0] - self._obj_min[0]) * self._inv_range[0]
        fn1 = (fv[1] - self._obj_min[1]) * self._inv_range[1]
        fn2 = (fv[2] - self._obj_min[2]) * self._inv_range[2]
        f_norm = (fn0, fn1, fn2)
        p1, p2, p3 = self.priority
        inv_tol = 1.0 / self.tol
        b1 = math.floor(f_norm[p1] * inv_tol)
        b2 = math.floor(f_norm[p2] * inv_tol)
        b3 = math.floor(f_norm[p3] * inv_tol)
        return (b1, b2, b3, f_norm[p1], f_norm[p2], f_norm[p3])

    def prepare_population(self, pop: list[Any]) -> list[Any]:
        for ind in pop:
            ind.lex_key = self._lex_key(ind)
        sorted_pop = sorted(pop, key=lambda ind: ind.lex_key)
        for rank, ind in enumerate(sorted_pop):
            ind.lex_rank = rank
        return sorted_pop

    def select_parents(self, pop: list[Any], k: int, rng: random.Random) -> list[Any]:
        """Torneo binario lexicográfico."""
        selected: list[Any] = []
        n = len(pop)
        for _ in range(k):
            i1, i2 = rng.randrange(n), rng.randrange(n)
            a, b = pop[i1], pop[i2]
            winner = a if a.lex_key <= b.lex_key else b
            selected.append(winner)
        return selected

    def select_survivors(self, combined_pop: list[Any], k: int) -> list[Any]:
        """Truncamiento elitista R_t = P_t U Q_t ordenando lexicográficamente."""
        for ind in combined_pop:
            ind.lex_key = self._lex_key(ind)
        survivors = sorted(combined_pop, key=lambda ind: ind.lex_key)[:k]
        for rank, ind in enumerate(survivors):
            ind.lex_rank = rank
        return survivors

    def get_representative_solution(self, pop: list[Any]) -> Any:
        """Devuelve la solución que encabeza la jerarquía lexicográfica."""
        return min(pop, key=lambda ind: getattr(ind, "lex_key", self._lex_key(ind)))


# ==============================================================================
# 2. ENFOQUE 2: SUMA PONDERADA NORMALIZADA
# ==============================================================================
class WeightedSumStrategy:
    """
    Escalariza los objetivos mediante normalización min-max previa y suma ponderada:
        F(x) = w1 * f1_norm(x) + w2 * f2_norm(x) + w3 * f3_norm(x), con sum(w) = 1.
    """

    def __init__(
        self,
        weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        name: str = "Pond_Equilibrada",
        label: str = "Suma Ponderada (1/3, 1/3, 1/3)",
    ) -> None:
        w_arr = np.asarray(weights, dtype=float)
        if not np.isclose(np.sum(w_arr), 1.0, atol=1e-6):
            w_arr = w_arr / np.sum(w_arr)
        self.weights = tuple(float(w) for w in w_arr)
        self._obj_min = tuple(float(v) for v in problem.OBJ_MIN)
        self._inv_range = tuple(
            1.0 / float(mx - mn) if (mx - mn) > 1e-12 else 1.0
            for mn, mx in zip(problem.OBJ_MIN, problem.OBJ_MAX)
        )
        self.name = name
        self.label = label

    def _scalar_value(self, ind: Any) -> float:
        fv = ind.fitness.values
        w0, w1, w2 = self.weights
        return (
            w0 * (fv[0] - self._obj_min[0]) * self._inv_range[0]
            + w1 * (fv[1] - self._obj_min[1]) * self._inv_range[1]
            + w2 * (fv[2] - self._obj_min[2]) * self._inv_range[2]
        )

    def prepare_population(self, pop: list[Any]) -> list[Any]:
        for ind in pop:
            ind.scalar_fitness = self._scalar_value(ind)
        return sorted(pop, key=lambda ind: ind.scalar_fitness)

    def select_parents(self, pop: list[Any], k: int, rng: random.Random) -> list[Any]:
        """Torneo binario por valor escalarizado ponderado."""
        selected: list[Any] = []
        n = len(pop)
        for _ in range(k):
            i1, i2 = rng.randrange(n), rng.randrange(n)
            a, b = pop[i1], pop[i2]
            winner = a if a.scalar_fitness <= b.scalar_fitness else b
            selected.append(winner)
        return selected

    def select_survivors(self, combined_pop: list[Any], k: int) -> list[Any]:
        """Truncamiento elitista R_t = P_t U Q_t conservando los k menores F(x)."""
        for ind in combined_pop:
            ind.scalar_fitness = self._scalar_value(ind)
        return sorted(combined_pop, key=lambda ind: ind.scalar_fitness)[:k]

    def get_representative_solution(self, pop: list[Any]) -> Any:
        """Devuelve la solución con menor valor escalarizado F(x)."""
        return min(pop, key=lambda ind: getattr(ind, "scalar_fitness", self._scalar_value(ind)))


# ==============================================================================
# 3. ENFOQUE 3: DOMINANCIA DE PARETO CON NSGA-II
# ==============================================================================
class NSGA2Strategy:
    """
    Aplica ordenamiento rápido no dominado (Frente 1, Frente 2, ...), distancia de
    hacinamiento (crowding distance) y truncamiento elitista mediante `deap.tools.selNSGA2`.
    """

    def __init__(
        self,
        name: str = "NSGA_II",
        label: str = "NSGA-II (Dominancia de Pareto)",
    ) -> None:
        self.name = name
        self.label = label

    def _assign_fronts_and_crowding(self, pop: list[Any]) -> list[Any]:
        """Asigna el número de frente (1, 2, ...) y la crowding distance usando DEAP."""
        pop_nsga = tools.selNSGA2(pop, len(pop))
        fronts = tools.sortNondominated(pop_nsga, len(pop_nsga))
        for rank_idx, front in enumerate(fronts, start=1):
            tools.emo.assignCrowdingDist(front)
            for ind in front:
                ind.front_rank = rank_idx
        return pop_nsga

    def prepare_population(self, pop: list[Any]) -> list[Any]:
        return self._assign_fronts_and_crowding(pop)

    def select_parents(self, pop: list[Any], k: int, rng: random.Random) -> list[Any]:
        """
        Torneo binario de hacinamiento (Crowded Binary Tournament):
        Prefiere menor rango de frente; en caso de empate, mayor crowding distance.
        """
        selected: list[Any] = []
        n = len(pop)
        for _ in range(k):
            i1, i2 = rng.randrange(n), rng.randrange(n)
            a, b = pop[i1], pop[i2]
            rank_a = getattr(a, "front_rank", 999)
            rank_b = getattr(b, "front_rank", 999)
            if rank_a < rank_b:
                winner = a
            elif rank_b < rank_a:
                winner = b
            else:
                cd_a = getattr(a.fitness, "crowding_dist", 0.0)
                cd_b = getattr(b.fitness, "crowding_dist", 0.0)
                winner = a if cd_a >= cd_b else b
            selected.append(winner)
        return selected

    def select_survivors(self, combined_pop: list[Any], k: int) -> list[Any]:
        """
        Truncamiento elitista R_t = P_t U Q_t con `deap.tools.selNSGA2`
        preservando diversidad sobre puntos únicos.
        """
        unique, duplicates = ga_core.deduplicate_population(combined_pop)
        if len(unique) >= k:
            survivors = tools.selNSGA2(unique, k)
        else:
            rem = tools.selNSGA2(duplicates, k - len(unique))
            survivors = unique + rem
        return self._assign_fronts_and_crowding(survivors)

    def get_representative_solution(self, pop: list[Any]) -> Any:
        """
        Del Frente 1 (no dominado), selecciona la solución compromiso de la región
        central (menor distancia euclidiana normalizada al punto ideal (0, 0, 0)),
        que representa una preferencia equilibrada a posteriori.
        """
        front1 = [ind for ind in pop if getattr(ind, "front_rank", 1) == 1]
        candidates = front1 if front1 else pop
        return min(
            candidates,
            key=lambda ind: float(np.linalg.norm(problem.normalize_objectives(ind.fitness.values))),
        )


# ==============================================================================
# 4. CONSTRUCTOR DE LAS CONFIGURACIONES EXPERIMENTALES
# ==============================================================================
def build_all_strategies(tol: float | None = None) -> list[Any]:
    """
    Construye las 6 configuraciones experimentales requeridas:
      1. Lexicográfica (F1 > F2 > F3)
      2. Lexicográfica permutada (F3 > F2 > F1) para evaluar sensibilidad al orden
      3. Suma Ponderada equilibrada (1/3, 1/3, 1/3)
      4. Suma Ponderada sesgada a F1 (0.70, 0.15, 0.15)
      5. Suma Ponderada sesgada a F3 (0.15, 0.15, 0.70)
      6. Dominancia de Pareto con NSGA-II
    """
    lex_tol = ga_core.GA_CONFIG["lex_tol"] if tol is None else tol
    return [
        LexicographicStrategy(
            priority=(0, 1, 2),
            tol=lex_tol,
            name="Lex_F1_F2_F3",
            label="Lexicográfica (F1 > F2 > F3)",
        ),
        LexicographicStrategy(
            priority=(2, 1, 0),
            tol=lex_tol,
            name="Lex_F3_F2_F1",
            label="Lexicográfica (F3 > F2 > F1)",
        ),
        WeightedSumStrategy(
            weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            name="Pond_Equilibrada",
            label="Suma Ponderada (1/3, 1/3, 1/3)",
        ),
        WeightedSumStrategy(
            weights=(0.70, 0.15, 0.15),
            name="Pond_Sesgo_F1",
            label="Suma Ponderada Sesgo F1 (0.70, 0.15, 0.15)",
        ),
        WeightedSumStrategy(
            weights=(0.15, 0.15, 0.70),
            name="Pond_Sesgo_F3",
            label="Suma Ponderada Sesgo F3 (0.15, 0.15, 0.70)",
        ),
        NSGA2Strategy(
            name="NSGA_II",
            label="NSGA-II (Dominancia de Pareto)",
        ),
    ]


if __name__ == "__main__":
    print("=== Prueba rápida de strategies.py (1 semilla, 100 generaciones) ===")
    for strat in build_all_strategies():
        res = ga_core.run_ga(strat, seed=101)
        rep = res["representative_objs"]
        hv_end = res["history"][-1]["hv_raw"]
        n_nd = len(res["nd_objs"])
        print(
            f"- {strat.label:42s} | "
            f"Compromiso f=({rep[0]:.4f}, {rep[1]:.4f}, {rep[2]:.4f}) | "
            f"NoDom={n_nd:3d} | HV={hv_end:.4f}"
        )
