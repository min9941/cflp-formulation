"""Constraint별 penalty 균형 실험 결과 시각화 (Figure 12 ~ 15).

축 레이블과 범례는 한글 폰트 의존성을 피하기 위해 영어로 작성한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODE_STYLE: dict[str, dict] = {
    "offline": {"color": "#e67e22", "marker": "o", "linestyle": "-"},
    "qpu": {"color": "#2980b9", "marker": "s", "linestyle": "--"},
}
ICE_LOW, ICE_HIGH = 0.01, 0.03


def _save(figure: plt.Figure, output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _series(results: pd.DataFrame) -> list[tuple[str, str, str, pd.DataFrame]]:
    """(instance, formulation, mode)별로 정렬된 그룹을 돌려준다."""
    groups = []
    for (instance, formulation, mode), group in results.groupby(
        ["instance", "formulation", "mode"], sort=False
    ):
        groups.append((instance, formulation, mode, group.sort_values("ratio")))
    return groups


def _mark_free_limit(axis: plt.Axes, results: pd.DataFrame) -> None:
    """coefficient range가 커지기 시작하는 ratio를 표시한다."""
    baseline = results[results["ratio"] == 1.0]
    if baseline.empty:
        return
    reference = float(baseline["qubo_range"].iloc[0])
    grown = results[results["qubo_range"] > reference * 1.01]
    if grown.empty:
        return
    limit = float(grown["ratio"].min())
    axis.axvline(
        limit, color="#c0392b", linestyle=":", linewidth=1.2,
        label="range starts growing",
    )


def plot_ratio_vs_feasibility(
    results: pd.DataFrame, output_dir: str | Path
) -> Path:
    """Figure 12 — assignment/capacity 비율과 feasible 비율."""
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for instance, formulation, mode, group in _series(results):
        style = MODE_STYLE.get(mode, MODE_STYLE["offline"])
        axis.plot(
            group["ratio"],
            group["feasible_fraction"] * 100.0,
            label=f"{instance} {formulation} ({mode})",
            **style,
        )
    _mark_free_limit(axis, results)
    axis.set_xscale("log")
    axis.set_xlabel("assignment penalty / capacity penalty")
    axis.set_ylabel("Feasible samples (%)")
    axis.set_title("Figure 12. Penalty balance vs feasible rate")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, output_dir, "figure12_balance_feasibility.png")


def plot_ratio_vs_gap(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 13 — 비율과 true optimality gap."""
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    plotted = False
    for instance, formulation, mode, group in _series(results):
        subset = group[np.isfinite(group["true_gap_percent"])]
        if subset.empty:
            continue
        plotted = True
        style = MODE_STYLE.get(mode, MODE_STYLE["offline"])
        axis.plot(
            subset["ratio"],
            subset["true_gap_percent"],
            label=f"{instance} {formulation} ({mode})",
            **style,
        )
    axis.axhline(0.0, color="#2c3e50", linewidth=1.0, label="Gurobi optimum (0%)")
    axis.set_xscale("log")
    axis.set_xlabel("assignment penalty / capacity penalty")
    axis.set_ylabel("True optimality gap (%)")
    axis.set_title("Figure 13. Penalty balance vs solution quality")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    if not plotted:
        axis.text(
            0.5, 0.5, "no feasible solution at any ratio",
            transform=axis.transAxes, ha="center", va="center", fontsize=11,
        )
    figure.tight_layout()
    return _save(figure, output_dir, "figure13_balance_gap.png")


def plot_constraint_strength(
    results: pd.DataFrame, output_dir: str | Path
) -> Path:
    """Figure 14 — 정규화된 제약 강도와 ICE 노이즈 대역.

    같은 lambda를 써도 assignment가 capacity보다 훨씬 약하다는 것,
    그리고 어느 ratio에서 노이즈 위로 올라오는지를 보여준다.
    """
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    first = True
    for instance, formulation, mode, group in _series(results):
        if mode != results["mode"].iloc[0]:
            continue
        axis.plot(
            group["ratio"], group["assignment_strength"],
            marker="o", color="#e67e22",
            label=f"{instance} {formulation} assignment" if first else None,
        )
        axis.plot(
            group["ratio"], group["capacity_strength"],
            marker="^", linestyle="--", color="#16a085",
            label=f"{instance} {formulation} capacity" if first else None,
        )
        axis.plot(
            group["ratio"], group["objective_strength"],
            marker="v", linestyle=":", color="#8e44ad",
            label=f"{instance} {formulation} objective" if first else None,
        )
        first = False
    axis.axhspan(
        ICE_LOW, ICE_HIGH, color="#7f8c8d", alpha=0.25,
        label=f"ICE noise ({ICE_LOW}-{ICE_HIGH})",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("assignment penalty / capacity penalty")
    axis.set_ylabel("Normalized coefficient (after auto-scale)")
    axis.set_title("Figure 14. Constraint strength vs hardware noise floor")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, output_dir, "figure14_constraint_strength.png")


def plot_ratio_vs_range(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 15 — 비율과 coefficient range.

    일정 ratio까지는 range가 전혀 변하지 않는다는 것, 즉 그 구간의 개선이
    **공짜**임을 보여주는 그림이다.
    """
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for instance, formulation, mode, group in _series(results):
        if mode != results["mode"].iloc[0]:
            continue
        axis.plot(
            group["ratio"], group["qubo_range"],
            marker="o", label=f"{instance} {formulation}",
        )
    _mark_free_limit(axis, results)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("assignment penalty / capacity penalty")
    axis.set_ylabel("QUBO coefficient range")
    axis.set_title("Figure 15. Penalty balance vs coefficient range")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, output_dir, "figure15_balance_range.png")


def generate_all(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Figure 12 ~ 15를 모두 생성한다."""
    return [
        plot_ratio_vs_feasibility(results, output_dir),
        plot_ratio_vs_gap(results, output_dir),
        plot_constraint_strength(results, output_dir),
        plot_ratio_vs_range(results, output_dir),
    ]
