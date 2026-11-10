"""
run_experiments.py
Ejecuta el diseño experimental completo:
  - 6 configuraciones de evaluación/selección (2 lexicográficas, 3 sumas ponderadas, 1 NSGA-II)
  - 10 semillas fijas y reportadas por configuración (60 corridas independientes)
  - Población = 100 individuos, Generaciones = 100

Guarda en la carpeta `results/`:
  1. `runs_summary.csv`        : Resumen de métricas por cada corrida y semilla.
  2. `final_populations.csv`   : Vectores de variables y objetivos de la población final.
  3. `convergence_history.csv` : Evolución por generación (hipervolumen y escalarizado).
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd

import problem
import ga_core
import strategies

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_all_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta todas las configuraciones y semillas y guarda los archivos CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = ga_core.GA_CONFIG
    seeds: list[int] = cfg["seeds"]
    strat_list = strategies.build_all_strategies(tol=cfg["lex_tol"])

    summary_rows: list[dict] = []
    pop_rows: list[dict] = []
    history_rows: list[dict] = []

    print("=" * 88)
    print(f"EJECUCIÓN EXPERIMENTAL MULTIOBJETIVO — {problem.PROBLEM_NAME}")
    print(
        f"Población: {cfg['pop_size']} | Generaciones: {cfg['n_gen']} | "
        f"Cruza: {cfg['cx_type'].upper()} (alpha={cfg['blx_alpha']}, p={cfg['cx_prob']}) | "
        f"Mutación: Gauss(sigma={cfg['mut_sigma']}, p_gen={cfg['mut_indpb']:.2f})"
    )
    print(f"Semillas fijas ({len(seeds)}): {seeds}")
    print(f"Punto de referencia Hipervolumen (original)    : {problem.HV_REF_POINT.tolist()}")
    print(f"Punto de referencia Hipervolumen (normalizado) : {problem.HV_REF_POINT_NORM.tolist()}")
    print("=" * 88)

    t0_global = time.perf_counter()

    for strat in strat_list:
        t0_strat = time.perf_counter()
        for seed in seeds:
            t0_run = time.perf_counter()
            res = ga_core.run_ga(strat, seed=seed, config=cfg)
            dt_run = time.perf_counter() - t0_run

            rep_ind = res["representative"]
            rep_f = res["representative_objs"]
            rep_f_norm = problem.normalize_objectives(rep_f)
            last_hist = res["history"][-1]

            # 1. Fila de resumen por corrida
            row_sum = {
                "strategy_name": strat.name,
                "strategy_label": strat.label,
                "seed": seed,
                "rep_f1": rep_f[0],
                "rep_f2": rep_f[1],
                "rep_f3": rep_f[2],
                "rep_f1_norm": float(rep_f_norm[0]),
                "rep_f2_norm": float(rep_f_norm[1]),
                "rep_f3_norm": float(rep_f_norm[2]),
                "n_nondominated": len(res["nd_objs"]),
                "hv_raw": last_hist["hv_raw"],
                "hv_norm": last_hist["hv_norm"],
                "best_equal_scalar": last_hist["best_equal_scalar"],
                "runtime_sec": dt_run,
            }
            for var_idx, var_name in enumerate(problem.VAR_NAMES):
                row_sum[f"rep_{var_name}"] = float(rep_ind[var_idx])
            summary_rows.append(row_sum)

            # 2. Filas de la población final
            final_pop = res["final_pop"]
            is_nd_mask = res["is_nd"]
            for idx, ind in enumerate(final_pop):
                f_vals = ind.fitness.values
                f_norm = problem.normalize_objectives(f_vals)
                is_rep = all(abs(a - b) < 1e-12 for a, b in zip(ind, rep_ind))
                row_pop = {
                    "strategy_name": strat.name,
                    "strategy_label": strat.label,
                    "seed": seed,
                    "ind_idx": idx,
                    "f1": f_vals[0],
                    "f2": f_vals[1],
                    "f3": f_vals[2],
                    "f1_norm": float(f_norm[0]),
                    "f2_norm": float(f_norm[1]),
                    "f3_norm": float(f_norm[2]),
                    "is_nondominated": bool(is_nd_mask[idx]),
                    "front_rank": int(getattr(ind, "front_rank", 1 if is_nd_mask[idx] else 2)),
                    "is_representative": bool(is_rep),
                }
                for var_idx, var_name in enumerate(problem.VAR_NAMES):
                    row_pop[var_name] = float(ind[var_idx])
                pop_rows.append(row_pop)

            # 3. Historial de convergencia por generación
            for h in res["history"]:
                history_rows.append(
                    {
                        "strategy_name": strat.name,
                        "strategy_label": strat.label,
                        "seed": seed,
                        **h,
                    }
                )

        dt_strat = time.perf_counter() - t0_strat
        df_s = pd.DataFrame([r for r in summary_rows if r["strategy_name"] == strat.name])
        print(
            f"[OK] {strat.label:42s} | "
            f"f1={df_s['rep_f1'].mean():.4f}±{df_s['rep_f1'].std():.4f}, "
            f"f2={df_s['rep_f2'].mean():.4f}±{df_s['rep_f2'].std():.4f}, "
            f"f3={df_s['rep_f3'].mean():.4f}±{df_s['rep_f3'].std():.4f} | "
            f"NoDom={df_s['n_nondominated'].mean():.1f} | "
            f"HV={df_s['hv_raw'].mean():.4f} ({dt_strat:.1f}s)"
        )

    df_summary = pd.DataFrame(summary_rows)
    df_pop = pd.DataFrame(pop_rows)
    df_history = pd.DataFrame(history_rows)

    summary_path = RESULTS_DIR / "runs_summary.csv"
    pop_path = RESULTS_DIR / "final_populations.csv"
    history_path = RESULTS_DIR / "convergence_history.csv"

    df_summary.to_csv(summary_path, index=False, encoding="utf-8")
    df_pop.to_csv(pop_path, index=False, encoding="utf-8")
    df_history.to_csv(history_path, index=False, encoding="utf-8")

    total_time = time.perf_counter() - t0_global
    print("-" * 88)
    print(f"Experimentos completados en {total_time:.2f} s.")
    print(f"Archivos guardados en: {RESULTS_DIR}")
    print(f"  - {summary_path.name} ({len(df_summary)} filas)")
    print(f"  - {pop_path.name} ({len(df_pop)} filas)")
    print(f"  - {history_path.name} ({len(df_history)} filas)")
    return df_summary, df_pop, df_history


if __name__ == "__main__":
    run_all_experiments()
