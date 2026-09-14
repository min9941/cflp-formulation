"""Penalty coefficient 실험 결과 시각화 (Figure 8 ~ 11).

축 레이블과 범례는 한글 폰트 의존성을 피하기 위해 영어로 작성한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SOLVER_COLORS: dict[str, str] = {"SA": "#e67e22", "QA": "#2980b9"}
BOUNDARY_COLOR = "#c0392b"
# ICE 노이즈의 대략적 범위(정규화 단위). 정확한 spec 값이 아니라 자릿수 비교용.
ICE_LOW, ICE_HIGH = 0.01, 0.03


def _save(figure: plt.Figure, output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _mark_boundary(axis: plt.Axes, label: bool = True) -> None:
    """이론적 경계 m = 1을 표시한다."""
    axis.axvline(
        1.0,
        color=BOUNDARY_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="theoretical bound (m = 1)" if label else None,
    )


def _shade_unsafe(axis: plt.Axes) -> None:
    """m < 1 구간은 이론적 보장이 없으므로 음영 처리한다."""
    left = axis.get_xlim()[0]
    axis.axvspan(left, 1.0, color=BOUNDARY_COLOR, alpha=0.06)


def plot_lambda_vs_range(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 8 — lambda와 QUBO 계수 범위의 관계."""
    frame = results.drop_duplicates(subset=["multiplier"]).sort_values("multiplier")
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(
        frame["multiplier"],
        frame["qubo_range"],
        marker="o",
        color="#8e44ad",
        label="QUBO coefficient range",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    _mark_boundary(axis)
    axis.set_xlabel("m = lambda / Z_ub")
    axis.set_ylabel("Coefficient range (max - min)")
    axis.set_title("Figure 8. Penalty coefficient vs QUBO coefficient range")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, output_dir, "figure8_lambda_range.png")


def plot_lambda_vs_feasibility(
    results: pd.DataFrame, output_dir: str | Path
) -> Path:
    """Figure 9 — lambda와 feasible 비율. 두 가설을 구분하는 핵심 그림."""
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for solver in ("SA", "QA"):
        subset = results[results["solver"] == solver].sort_values("multiplier")
        if subset.empty:
            continue
        axis.plot(
            subset["multiplier"],
            subset["feasible_fraction"] * 100.0,
            marker="o" if solver == "SA" else "s",
            color=SOLVER_COLORS[solver],
            label=solver,
        )
    axis.set_xscale("log")
    _mark_boundary(axis)
    _shade_unsafe(axis)
    axis.set_xlabel("m = lambda / Z_ub   (shaded area: no feasibility guarantee)")
    axis.set_ylabel("Feasible samples (%)")
    axis.set_title("Figure 9. Penalty coefficient vs feasible rate")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, output_dir, "figure9_lambda_feasibility.png")


def plot_lambda_vs_gap(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 10 — lambda와 true optimality gap."""
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    plotted = False
    for solver in ("SA", "QA"):
        subset = results[results["solver"] == solver].sort_values("multiplier")
        subset = subset[np.isfinite(subset["true_gap_percent"])]
        if subset.empty:
            continue
        plotted = True
        axis.plot(
            subset["multiplier"],
            subset["true_gap_percent"],
            marker="o" if solver == "SA" else "s",
            color=SOLVER_COLORS[solver],
            label=solver,
        )
    axis.axhline(0.0, color="#2c3e50", linewidth=1.0, label="Gurobi optimum (0%)")
    axis.set_xscale("log")
    _mark_boundary(axis, label=False)
    _shade_unsafe(axis)
    axis.set_xlabel("m = lambda / Z_ub")
    axis.set_ylabel("True optimality gap (%)")
    axis.set_title("Figure 10. Penalty coefficient vs solution quality")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    if not plotted:
        axis.text(
            0.5, 0.5,
            "no feasible solution at any lambda",
            transform=axis.transAxes, ha="center", va="center", fontsize=11,
        )
    figure.tight_layout()
    return _save(figure, output_dir, "figure10_lambda_gap.png")


def plot_scaled_objective(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 11 — auto_scale 이후 objective 계수와 ICE 노이즈 비교."""
    frame = results.drop_duplicates(subset=["multiplier"]).sort_values("multiplier")
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(
        frame["multiplier"],
        frame["scaled_objective_coefficient"],
        marker="o",
        color="#16a085",
        label="smallest objective coefficient after auto-scale",
    )
    axis.axhspan(
        ICE_LOW, ICE_HIGH, color="#7f8c8d", alpha=0.25,
        label=f"ICE noise level ({ICE_LOW}-{ICE_HIGH})",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    _mark_boundary(axis, label=False)
    axis.set_xlabel("m = lambda / Z_ub")
    axis.set_ylabel("Normalized coefficient")
    axis.set_title("Figure 11. Objective signal vs hardware noise floor")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    return _save(figure, output_dir, "figure11_objective_vs_noise.png")


def generate_all(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Figure 8 ~ 11을 모두 생성한다."""
    return [
        plot_lambda_vs_range(results, output_dir),
        plot_lambda_vs_feasibility(results, output_dir),
        plot_lambda_vs_gap(results, output_dir),
        plot_scaled_objective(results, output_dir),
    ]
