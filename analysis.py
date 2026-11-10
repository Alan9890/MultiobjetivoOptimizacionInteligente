"""
analysis.py
Calcula las métricas estadísticas sobre las 10 semillas por configuración, el indicador
de sesgo respecto a NSGA-II y respecto a la solución equilibrada, y genera todas las
gráficas requeridas en alta resolución (200 dpi) dentro de `results/`.

Terminología aplicada en tablas y gráficas:
  Se evita la expresión "solución óptima"; se emplea "solución que entrega un buen
  compromiso según las preferencias del tomador de decisiones".
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import problem
import ga_core

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Paleta de colores distinguible y consistente para las 6 configuraciones
STRATEGY_ORDER = [
    "Lex_F1_F2_F3",
    "Lex_F3_F2_F1",
    "Pond_Equilibrada",
    "Pond_Sesgo_F1",
    "Pond_Sesgo_F3",
    "NSGA_II",
]

STRATEGY_COLORS = {
    "Lex_F1_F2_F3": "#d95f02",      # Naranja intenso
    "Lex_F3_F2_F1": "#e7298a",      # Magenta
    "Pond_Equilibrada": "#7570b3",  # Violeta
    "Pond_Sesgo_F1": "#66a61e",     # Verde oliva
    "Pond_Sesgo_F3": "#e6ab02",     # Dorado
    "NSGA_II": "#1b9e77",           # Verde azulado (Teal)
}

STRATEGY_MARKERS = {
    "Lex_F1_F2_F3": "^",
    "Lex_F3_F2_F1": "v",
    "Pond_Equilibrada": "s",
    "Pond_Sesgo_F1": "D",
    "Pond_Sesgo_F3": "P",
    "NSGA_II": "o",
}

STRATEGY_SHORT_LABELS = {
    "Lex_F1_F2_F3": "Lex (F1>F2>F3)",
    "Lex_F3_F2_F1": "Lex (F3>F2>F1)",
    "Pond_Equilibrada": "Pond. (1/3, 1/3, 1/3)",
    "Pond_Sesgo_F1": "Pond. Sesgo F1 (0.70)",
    "Pond_Sesgo_F3": "Pond. Sesgo F3 (0.70)",
    "NSGA_II": "NSGA-II (Pareto)",
}


def compute_metrics_table(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los resultados por método sobre las 10 semillas y calcula:
      - Media y desviación estándar de f1, f2, f3 de la solución compromiso representativa.
      - Número medio (± std) de soluciones no dominadas encontradas.
      - Hipervolumen medio (± std) en espacio original y normalizado.
      - Indicador de sesgo (% de cambio en cada objetivo respecto a NSGA-II y respecto
        a la Suma Ponderada equilibrada). Un valor positivo indica reducción (mejora)
        en ese objetivo; un valor negativo indica incremento (sacrificio/deterioro).
    """
    rows: list[dict] = []

    # Valores de referencia para el indicador de sesgo
    nsga_df = df_summary[df_summary["strategy_name"] == "NSGA_II"]
    eq_df = df_summary[df_summary["strategy_name"] == "Pond_Equilibrada"]

    ref_nsga = np.array([
        nsga_df["rep_f1"].mean(),
        nsga_df["rep_f2"].mean(),
        nsga_df["rep_f3"].mean(),
    ])
    ref_eq = np.array([
        eq_df["rep_f1"].mean(),
        eq_df["rep_f2"].mean(),
        eq_df["rep_f3"].mean(),
    ])

    for s_name in STRATEGY_ORDER:
        sub = df_summary[df_summary["strategy_name"] == s_name]
        if sub.empty:
            continue
        label = sub["strategy_label"].iloc[0]

        f1_m, f1_s = sub["rep_f1"].mean(), sub["rep_f1"].std()
        f2_m, f2_s = sub["rep_f2"].mean(), sub["rep_f2"].std()
        f3_m, f3_s = sub["rep_f3"].mean(), sub["rep_f3"].std()

        nd_m, nd_s = sub["n_nondominated"].mean(), sub["n_nondominated"].std()
        hv_m, hv_s = sub["hv_raw"].mean(), sub["hv_raw"].std()
        hvn_m, hvn_s = sub["hv_norm"].mean(), sub["hv_norm"].std()
        eq_m, eq_s = sub["best_equal_scalar"].mean(), sub["best_equal_scalar"].std()

        # Indicador de sesgo porcentual respecto a NSGA-II:
        # Mejora (%) = (f_ref - f_metodo) / f_ref * 100
        # Positivo = reduce (mejora) ese objetivo respecto al compromiso central de NSGA-II.
        # Negativo = aumenta (sacrifica) ese objetivo respecto al compromiso central de NSGA-II.
        mejora_vs_nsga = (ref_nsga - np.array([f1_m, f2_m, f3_m])) / ref_nsga * 100.0
        mejora_vs_eq = (ref_eq - np.array([f1_m, f2_m, f3_m])) / ref_eq * 100.0

        rows.append(
            {
                "strategy_name": s_name,
                "strategy_label": label,
                "f1_mean": f1_m,
                "f1_std": f1_s,
                "f2_mean": f2_m,
                "f2_std": f2_s,
                "f3_mean": f3_m,
                "f3_std": f3_s,
                "n_nd_mean": nd_m,
                "n_nd_std": nd_s,
                "hv_raw_mean": hv_m,
                "hv_raw_std": hv_s,
                "hv_norm_mean": hvn_m,
                "hv_norm_std": hvn_s,
                "equal_scalar_mean": eq_m,
                "equal_scalar_std": eq_s,
                "sesgo_mejora_f1_vs_nsga_pct": mejora_vs_nsga[0],
                "sesgo_mejora_f2_vs_nsga_pct": mejora_vs_nsga[1],
                "sesgo_mejora_f3_vs_nsga_pct": mejora_vs_nsga[2],
                "sesgo_mejora_f1_vs_eq_pct": mejora_vs_eq[0],
                "sesgo_mejora_f2_vs_eq_pct": mejora_vs_eq[1],
                "sesgo_mejora_f3_vs_eq_pct": mejora_vs_eq[2],
                "s1_mean": sub["rep_s1"].mean(),
                "s2_mean": sub["rep_s2"].mean(),
                "s3_mean": sub["rep_s3"].mean(),
                "s4_mean": sub["rep_s4"].mean(),
                "v_mean": sub["rep_v"].mean(),
            }
        )

    df_metrics = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "metrics_comparison.csv"
    df_metrics.to_csv(out_csv, index=False, encoding="utf-8")
    return df_metrics


def plot_fig1_3d_and_2d_projections(df_pop: pd.DataFrame, df_summary: pd.DataFrame) -> Path:
    """
    Gráfica 1: Dispersión 3D y 3 proyecciones 2D del espacio de objetivos,
    mismo rango de ejes, un color por método. Muestra las soluciones no dominadas
    de NSGA-II (semilla representativa) y las soluciones compromiso de cada método
    en las 10 semillas.
    """
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Espacio de Objetivos (3D y Proyecciones 2D): Soluciones con Buen Compromiso\n"
        "según las Preferencias del Tomador de Decisiones (10 semillas por configuración)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Rangos fijos comunes para los 3 objetivos
    xlim = (-0.15, 5.8)
    ylim = (-0.10, 3.35)
    zlim = (-0.05, 1.38)

    # Subconjunto de NSGA-II (soluciones no dominadas de la semilla 101 para poblar la superficie)
    nsga_nd = df_pop[
        (df_pop["strategy_name"] == "NSGA_II")
        & (df_pop["seed"] == ga_core.GA_CONFIG["seeds"][0])
        & (df_pop["is_nondominated"] == True)
    ]

    # 1. Panel 3D
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(
        nsga_nd["f1"],
        nsga_nd["f2"],
        nsga_nd["f3"],
        c=STRATEGY_COLORS["NSGA_II"],
        marker="o",
        s=28,
        alpha=0.35,
        label="NSGA-II (abanico Frente 1, sem. 101)",
    )
    for s_name in STRATEGY_ORDER:
        sub = df_summary[df_summary["strategy_name"] == s_name]
        ax3d.scatter(
            sub["rep_f1"],
            sub["rep_f2"],
            sub["rep_f3"],
            c=STRATEGY_COLORS[s_name],
            marker=STRATEGY_MARKERS[s_name],
            s=75,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            label=STRATEGY_SHORT_LABELS[s_name],
        )
    ax3d.set_xlim(*xlim)
    ax3d.set_ylim(*ylim)
    ax3d.set_zlim(*zlim)
    ax3d.set_xlabel(problem.OBJ_NAMES[0], labelpad=6)
    ax3d.set_ylabel(problem.OBJ_NAMES[1], labelpad=6)
    ax3d.set_zlabel(problem.OBJ_NAMES[2], labelpad=6)
    ax3d.set_title("A) Dispersión 3D en el Espacio de Objetivos (f1, f2, f3)", fontweight="bold")
    ax3d.view_init(elev=24, azim=42)
    ax3d.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Proyecciones 2D: (f1, f2), (f1, f3), (f2, f3)
    projections = [
        (2, "rep_f1", "rep_f2", "f1", "f2", problem.OBJ_NAMES[0], problem.OBJ_NAMES[1], xlim, ylim, "B) Proyección 2D: f1 vs f2 (Defectos escapados vs Falsas alarmas)"),
        (3, "rep_f1", "rep_f3", "f1", "f3", problem.OBJ_NAMES[0], problem.OBJ_NAMES[2], xlim, zlim, "C) Proyección 2D: f1 vs f3 (Defectos escapados vs Tiempo/costo)"),
        (4, "rep_f2", "rep_f3", "f2", "f3", problem.OBJ_NAMES[1], problem.OBJ_NAMES[2], ylim, zlim, "D) Proyección 2D: f2 vs f3 (Falsas alarmas vs Tiempo/costo)"),
    ]

    for panel_idx, rx, ry, px, py, xlabel, ylabel, lim_x, lim_y, title in projections:
        ax = fig.add_subplot(2, 2, panel_idx)
        ax.scatter(
            nsga_nd[px],
            nsga_nd[py],
            c=STRATEGY_COLORS["NSGA_II"],
            marker="o",
            s=30,
            alpha=0.30,
            label="NSGA-II (Frente 1, sem. 101)",
        )
        for s_name in STRATEGY_ORDER:
            sub = df_summary[df_summary["strategy_name"] == s_name]
            ax.scatter(
                sub[rx],
                sub[ry],
                c=STRATEGY_COLORS[s_name],
                marker=STRATEGY_MARKERS[s_name],
                s=75,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.92,
                label=STRATEGY_SHORT_LABELS[s_name],
            )
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if panel_idx == 2:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    out_path = RESULTS_DIR / "fig1_espacio_objetivos_3d_2d.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_fig2_nsga2_pareto_front(df_pop: pd.DataFrame) -> Path:
    """
    Gráfica 2: Frente de Pareto aproximado por NSGA-II, resaltando Frente 1 (soluciones
    no dominadas), soluciones de frentes secundarios/iniciales y las Regiones de Interés (ROI)
    para distintos perfiles del tomador de decisiones.
    """
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(
        "Aproximación del Frente de Pareto con NSGA-II: Resaltando el Frente 1 (No Dominado)\n"
        "y las Regiones de Interés (ROI) según las Preferencias del Tomador de Decisiones",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    # Tomamos todas las soluciones de NSGA-II en las 10 semillas
    nsga_all = df_pop[df_pop["strategy_name"] == "NSGA_II"].copy()
    pts_all = nsga_all[["f1", "f2", "f3"]].to_numpy()

    # Clasificar en Frente 1 global consolidado vs soluciones dominadas entre semillas
    is_global_f1, _ = ga_core.get_nondominated_vectors(pts_all, tol_decimals=6)
    nsga_all["es_frente1_global"] = is_global_f1

    f1_df = nsga_all[nsga_all["es_frente1_global"] == True]
    dom_df = nsga_all[nsga_all["es_frente1_global"] == False]

    # También generamos una muestra aleatoria inicial (P_0) para contraste visual de dominancia
    rng = np.random.default_rng(101)
    p0_vars = rng.uniform(0.0, 1.0, size=(120, problem.N_VARS))
    p0_objs = np.array([problem.evaluate(x) for x in p0_vars])

    # Identificar 3 puntos de compromiso (ROI) dentro del Frente 1:
    # 1) ROI Prioridad Calidad (mínimo f1)
    idx_roi_f1 = f1_df["f1"].idxmin()
    # 2) ROI Prioridad Velocidad/Costo (mínimo f3)
    idx_roi_f3 = f1_df["f3"].idxmin()
    # 3) ROI Compromiso Equilibrado (menor distancia normalizada al ideal)
    norms = np.linalg.norm(f1_df[["f1_norm", "f2_norm", "f3_norm"]].to_numpy(), axis=1)
    idx_roi_eq = f1_df.index[int(np.argmin(norms))]

    # Panel izquierdo: Vista 3D del Frente 1 resaltado
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        p0_objs[:, 0],
        p0_objs[:, 1],
        p0_objs[:, 2],
        c="#b0b0b0",
        marker="x",
        s=22,
        alpha=0.45,
        label="Población inicial P0 (dominada / Frentes ≥ 2)",
    )
    if len(dom_df) > 0:
        ax1.scatter(
            dom_df["f1"],
            dom_df["f2"],
            dom_df["f3"],
            c="#9ecae1",
            marker="o",
            s=22,
            alpha=0.45,
            label=f"NSGA-II Subfrentes locales (n={len(dom_df)})",
        )
    sc = ax1.scatter(
        f1_df["f1"],
        f1_df["f2"],
        f1_df["f3"],
        c=f1_df["f3"],
        cmap="viridis",
        marker="o",
        s=38,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
        label=f"Frente 1 No Dominado (n={len(f1_df)})",
    )

    # Resaltar los 3 compromisos de referencia (ROI)
    rois = [
        (f1_df.loc[idx_roi_f1], "#d95f02", "*", "ROI A: Prioridad Cero Defectos (bajo f1)"),
        (f1_df.loc[idx_roi_eq], "#e7298a", "D", "ROI B: Compromiso Central Equilibrado"),
        (f1_df.loc[idx_roi_f3], "#e6ab02", "P", "ROI C: Prioridad Alta Velocidad (bajo f3)"),
    ]
    for row_roi, col, mk, lbl in rois:
        ax1.scatter(
            [row_roi["f1"]],
            [row_roi["f2"]],
            [row_roi["f3"]],
            c=col,
            marker=mk,
            s=160,
            edgecolors="black",
            linewidths=1.2,
            zorder=10,
            label=lbl,
        )

    ax1.set_xlabel(problem.OBJ_NAMES[0], labelpad=6)
    ax1.set_ylabel(problem.OBJ_NAMES[1], labelpad=6)
    ax1.set_zlabel(problem.OBJ_NAMES[2], labelpad=6)
    ax1.set_title("A) Frente 1 de NSGA-II en 3D vs Soluciones Dominadas", fontweight="bold")
    ax1.view_init(elev=25, azim=45)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel derecho: Proyección f1 vs f2 coloreada por f3 y mostrando el Frente 1
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(
        p0_objs[:, 0],
        p0_objs[:, 1],
        c="#b0b0b0",
        marker="x",
        s=25,
        alpha=0.5,
        label="Soluciones dominadas (P0 / Frentes secundarios)",
    )
    sc2 = ax2.scatter(
        f1_df["f1"],
        f1_df["f2"],
        c=f1_df["f3"],
        cmap="viridis",
        s=42,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
        label="Frente 1 (No Dominado — Compromisos de Pareto)",
    )
    cbar = fig.colorbar(sc2, ax=ax2, pad=0.02)
    cbar.set_label(problem.OBJ_NAMES[2])

    for row_roi, col, mk, lbl in rois:
        ax2.scatter(
            [row_roi["f1"]],
            [row_roi["f2"]],
            c=col,
            marker=mk,
            s=170,
            edgecolors="black",
            linewidths=1.2,
            zorder=10,
            label=lbl,
        )

    ax2.set_xlabel(problem.OBJ_NAMES[0])
    ax2.set_ylabel(problem.OBJ_NAMES[1])
    ax2.set_title("B) Mapa de Compromisos en Frente 1 (color = f3) y Regiones de Interés", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out_path = RESULTS_DIR / "fig2_frente_pareto_nsga2.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_fig3_bars_by_objective(df_metrics: pd.DataFrame) -> Path:
    """
    Gráfica 3: Barras por objetivo comparando métodos (media ± desviación estándar
    sobre las 10 semillas), más comparativa de Hipervolumen y Soluciones No Dominadas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Comparación Cuantitativa por Método (Media ± Desviación Estándar en 10 Semillas)\n"
        "Calidad del Compromiso Entregado según las Preferencias del Tomador de Decisiones",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(df_metrics))
    short_labels = [STRATEGY_SHORT_LABELS[s] for s in df_metrics["strategy_name"]]
    colors = [STRATEGY_COLORS[s] for s in df_metrics["strategy_name"]]

    panels = [
        (axes[0, 0], "f1_mean", "f1_std", problem.OBJ_NAMES[0], "A) Objetivo f1: Defectos que se escapan (menor = mejor)"),
        (axes[0, 1], "f2_mean", "f2_std", problem.OBJ_NAMES[1], "B) Objetivo f2: Falsas alarmas (menor = mejor)"),
        (axes[1, 0], "f3_mean", "f3_std", problem.OBJ_NAMES[2], "C) Objetivo f3: Tiempo/costo de inspección (menor = mejor)"),
        (axes[1, 1], "hv_raw_mean", "hv_raw_std", "Hipervolumen 3D [unid.³]", "D) Hipervolumen del Conjunto No Dominado (mayor = mayor cobertura)"),
    ]

    for ax, mean_col, std_col, ylabel, title in panels:
        means = df_metrics[mean_col].to_numpy()
        stds = df_metrics[std_col].to_numpy()
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.88,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=22, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Anotar valores sobre cada barra
        ymax = max(np.max(means + stds) * 1.18, 0.1)
        ax.set_ylim(0, ymax)
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                m + s + ymax * 0.02,
                f"{m:.3f}\n(±{s:.3f})",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out_path = RESULTS_DIR / "fig3_barras_objetivos.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_fig4_convergence(df_history: pd.DataFrame) -> Path:
    """
    Gráfica 4: Convergencia por generación (0 a 100) mostrando:
      - Panel A: Evolución del Hipervolumen 3D de la población Pt (media ± 1 desv. est.).
      - Panel B: Convergencia monótona (best-so-far) de la función de calidad normalizada
        propia de cada configuración en [0, 1].
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    fig.suptitle(
        "Convergencia Evolutiva por Generación (Media ± 1 Desv. Est. sobre 10 Semillas)\n"
        "Evolución hacia Soluciones que Entregan un Buen Compromiso",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    # Calcular la métrica escalarizada propia de cada método en cada generación y semilla
    df_h = df_history.copy()
    fn1 = df_h["rep_f1"] / float(problem.OBJ_MAX[0])
    fn2 = df_h["rep_f2"] / float(problem.OBJ_MAX[1])
    fn3 = df_h["rep_f3"] / float(problem.OBJ_MAX[2])

    own_scalar = np.zeros(len(df_h), dtype=float)
    for idx_s, s_name in enumerate(STRATEGY_ORDER):
        mask = df_h["strategy_name"] == s_name
        if s_name == "Lex_F1_F2_F3":
            own_scalar[mask] = fn1[mask] + 1e-3 * fn2[mask] + 1e-6 * fn3[mask]
        elif s_name == "Lex_F3_F2_F1":
            own_scalar[mask] = fn3[mask] + 1e-3 * fn2[mask] + 1e-6 * fn1[mask]
        elif s_name == "Pond_Equilibrada":
            own_scalar[mask] = (fn1[mask] + fn2[mask] + fn3[mask]) / 3.0
        elif s_name == "Pond_Sesgo_F1":
            own_scalar[mask] = 0.70 * fn1[mask] + 0.15 * fn2[mask] + 0.15 * fn3[mask]
        elif s_name == "Pond_Sesgo_F3":
            own_scalar[mask] = 0.15 * fn1[mask] + 0.15 * fn2[mask] + 0.70 * fn3[mask]
        else:  # NSGA_II
            own_scalar[mask] = df_h.loc[mask, "best_equal_scalar"]

    df_h["own_scalar"] = own_scalar
    # Best-so-far acumulado por corrida (semilla y estrategia)
    df_h = df_h.sort_values(["strategy_name", "seed", "gen"])
    df_h["own_scalar_bsf"] = df_h.groupby(["strategy_name", "seed"])["own_scalar"].cummin()

    grouped = (
        df_h.groupby(["strategy_name", "gen"])
        .agg(
            hv_mean=("hv_raw", "mean"),
            hv_std=("hv_raw", "std"),
            sc_mean=("own_scalar_bsf", "mean"),
            sc_std=("own_scalar_bsf", "std"),
        )
        .reset_index()
    )

    for s_name in STRATEGY_ORDER:
        sub = grouped[grouped["strategy_name"] == s_name].sort_values("gen")
        gens = sub["gen"].to_numpy()
        col = STRATEGY_COLORS[s_name]
        lbl = STRATEGY_SHORT_LABELS[s_name]

        # Panel A: Hipervolumen
        hv_m = sub["hv_mean"].to_numpy()
        hv_s = sub["hv_std"].fillna(0.0).to_numpy()
        ax1.plot(gens, hv_m, color=col, linewidth=2.2, label=lbl)
        ax1.fill_between(gens, hv_m - hv_s, hv_m + hv_s, color=col, alpha=0.15)

        # Panel B: Valor escalarizado propio (best-so-far)
        sc_m = sub["sc_mean"].to_numpy()
        sc_s = sub["sc_std"].fillna(0.0).to_numpy()
        ax2.plot(gens, sc_m, color=col, linewidth=2.2, label=lbl)
        ax2.fill_between(gens, sc_m - sc_s, sc_m + sc_s, color=col, alpha=0.15)

    ax1.set_xlabel("Generación (t)")
    ax1.set_ylabel("Hipervolumen 3D respecto a [6.0, 3.5, 1.5] [unid.³]")
    ax1.set_title("A) Evolución del Hipervolumen 3D de la Población", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="center right", fontsize=8.5, framealpha=0.92)

    ax2.set_xlabel("Generación (t)")
    ax2.set_ylabel("Mejor Valor Escalarizado Normalizado [adim. 0–1]")
    ax2.set_title("B) Convergencia del Mejor Valor Escalarizado (Best-so-Far)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.92])
    out_path = RESULTS_DIR / "fig4_convergencia_generaciones.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_fig5_bias_and_tradeoff(df_metrics: pd.DataFrame) -> Path:
    """
    Gráfica 5 (complementaria para la presentación):
      - Panel A: Indicador de sesgo (% de mejora [+] o sacrificio [-] en cada objetivo
        respecto al compromiso central de NSGA-II).
      - Panel B: Perfil de coordenadas paralelas normalizadas [0, 1] mostrando el
        trade-off (compromiso) entre f1, f2 y f3 para cada enfoque.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    fig.suptitle(
        "Análisis de Sesgo y Perfiles de Compromiso (Trade-off) entre Objetivos\n"
        "Efecto de la Jerarquía de Prioridades y los Pesos frente a la Dominancia de Pareto",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Barras agrupadas de sesgo respecto a NSGA-II
    sub_m = df_metrics[df_metrics["strategy_name"] != "NSGA_II"].copy()
    y_pos = np.arange(len(sub_m))
    height = 0.25

    b_f1 = sub_m["sesgo_mejora_f1_vs_nsga_pct"].to_numpy()
    b_f2 = sub_m["sesgo_mejora_f2_vs_nsga_pct"].to_numpy()
    b_f3 = sub_m["sesgo_mejora_f3_vs_nsga_pct"].to_numpy()
    labels = [STRATEGY_SHORT_LABELS[s] for s in sub_m["strategy_name"]]

    ax1.barh(y_pos + height, b_f1, height=height, color="#2b5c8f", edgecolor="black", label="f1 (Defectos escapados)")
    ax1.barh(y_pos, b_f2, height=height, color="#d95f02", edgecolor="black", label="f2 (Falsas alarmas)")
    ax1.barh(y_pos - height, b_f3, height=height, color="#1b9e77", edgecolor="black", label="f3 (Tiempo/costo)")
    ax1.axvline(0.0, color="black", linewidth=1.1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.set_xlabel("Cambio porcentual respecto a NSGA-II [%]\n(+) Mejora el objetivo  |  (−) Sacrifica el objetivo")
    ax1.set_title("A) Indicador de Sesgo respecto al Compromiso Central de NSGA-II", fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    # Panel B: Coordenadas paralelas normalizadas (f1_norm, f2_norm, f3_norm)
    axes_x = [0, 1, 2]
    axes_names = [
        "f1_norm\n(Defectos escapados)",
        "f2_norm\n(Falsas alarmas)",
        "f3_norm\n(Tiempo/costo)",
    ]
    for _, row in df_metrics.iterrows():
        s_name = row["strategy_name"]
        vals_norm = [
            row["f1_mean"] / float(problem.OBJ_MAX[0]),
            row["f2_mean"] / float(problem.OBJ_MAX[1]),
            row["f3_mean"] / float(problem.OBJ_MAX[2]),
        ]
        ax2.plot(
            axes_x,
            vals_norm,
            marker=STRATEGY_MARKERS[s_name],
            markersize=8,
            linewidth=2.4,
            color=STRATEGY_COLORS[s_name],
            label=STRATEGY_SHORT_LABELS[s_name],
        )

    ax2.set_xticks(axes_x)
    ax2.set_xticklabels(axes_names, fontsize=9.5)
    ax2.set_ylim(-0.04, 1.05)
    ax2.set_ylabel("Valor Normalizado del Objetivo [0 = mínimo ideal, 1 = máximo peor]")
    ax2.set_title("B) Perfil de Compromiso (Trade-off) en Espacio Normalizado [0, 1]", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper center", ncol=2, fontsize=8.3, framealpha=0.92)

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.92])
    out_path = RESULTS_DIR / "fig5_indicador_sesgo_tradeoff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_analysis() -> pd.DataFrame:
    """Ejecuta el análisis completo a partir de los CSV generados en `results/`."""
    summary_path = RESULTS_DIR / "runs_summary.csv"
    pop_path = RESULTS_DIR / "final_populations.csv"
    history_path = RESULTS_DIR / "convergence_history.csv"

    if not (summary_path.exists() and pop_path.exists() and history_path.exists()):
        raise FileNotFoundError(
            "No se encontraron los archivos CSV en `results/`. Ejecuta primero `python run_experiments.py`."
        )

    df_summary = pd.read_csv(summary_path)
    df_pop = pd.read_csv(pop_path)
    df_history = pd.read_csv(history_path)

    df_metrics = compute_metrics_table(df_summary)

    p1 = plot_fig1_3d_and_2d_projections(df_pop, df_summary)
    p2 = plot_fig2_nsga2_pareto_front(df_pop)
    p3 = plot_fig3_bars_by_objective(df_metrics)
    p4 = plot_fig4_convergence(df_history)
    p5 = plot_fig5_bias_and_tradeoff(df_metrics)

    print("=" * 95)
    print("TABLA RESUMEN DE MÉTRICAS Y COMPARACIÓN (10 SEMILLAS POR MÉTODO)")
    print("=" * 95)
    for _, r in df_metrics.iterrows():
        print(
            f"- {r['strategy_label']:42s} | "
            f"f1={r['f1_mean']:.4f}±{r['f1_std']:.4f} | "
            f"f2={r['f2_mean']:.4f}±{r['f2_std']:.4f} | "
            f"f3={r['f3_mean']:.4f}±{r['f3_std']:.4f} | "
            f"NoDom={r['n_nd_mean']:5.1f}±{r['n_nd_std']:4.1f} | "
            f"HV={r['hv_raw_mean']:6.3f}±{r['hv_raw_std']:5.3f}"
        )

    print("\nINDICADOR DE SESGO (% DE MEJORA [+] O SACRIFICIO [-] RESPECTO AL COMPROMISO DE NSGA-II):")
    for _, r in df_metrics.iterrows():
        print(
            f"- {r['strategy_label']:42s} | "
            f"Sesgo f1: {r['sesgo_mejora_f1_vs_nsga_pct']:+7.1f}% | "
            f"Sesgo f2: {r['sesgo_mejora_f2_vs_nsga_pct']:+7.1f}% | "
            f"Sesgo f3: {r['sesgo_mejora_f3_vs_nsga_pct']:+7.1f}%"
        )

    print("\nGráficas generadas a 200 dpi en `results/`:")
    for p in (p1, p2, p3, p4, p5):
        print(f"  - {p.name}")

    return df_metrics


if __name__ == "__main__":
    run_analysis()
