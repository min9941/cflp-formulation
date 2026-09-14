"""결과 시각화 모듈 (Figure 1 ~ Figure 7).

figure의 축 레이블과 범례는 한글 폰트 의존성을 피하기 위해 영어로 작성한다.
(설명과 주석은 한국어를 사용한다.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FORMULATIONS: tuple[str, ...] = ("SS", "MS")
SOLVER_ORDER: tuple[str, ...] = ("Gurobi", "SA", "QA")
SOLVER_COLORS: dict[str, str] = {
    "Gurobi": "#2c3e50",
    "SA": "#e67e22",
    "QA": "#2980b9",
}
FORMULATION_COLORS: dict[str, str] = {"SS": "#8e44ad", "MS": "#16a085"}


def _ordered_instances(frame: pd.DataFrame) -> list[str]:
    """instance를 크기 순으로 정렬한 목록을 반환한다."""
    pairs = (
        frame[["instance", "size"]].drop_duplicates().sort_values("size")
    )
    return pairs["instance"].tolist()


def _save(figure: plt.Figure, output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _bar_positions(
    num_groups: int, num_series: int, width: float = 0.8
) -> tuple[np.ndarray, float]:
    """grouped bar chart의 x 위치와 막대 폭을 계산한다."""
    centers = np.arange(num_groups, dtype=float)
    bar_width = width / max(num_series, 1)
    return centers, bar_width


def plot_objective_comparison(
    results: pd.DataFrame, output_dir: str | Path
) -> Path:
    """Figure 1 — solver별 원래 CFLP objective 비교.

    QA가 실행 불가능한 경우 해당 막대는 표시하지 않는다.
    """
    instances = _ordered_instances(results)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for axis, formulation in zip(axes, FORMULATIONS):
        subset = results[results["formulation"] == formulation]
        centers, bar_width = _bar_positions(len(instances), len(SOLVER_ORDER))
        for offset, solver in enumerate(SOLVER_ORDER):
            values = []
            positions = []
            for index, instance in enumerate(instances):
                row = subset[
                    (subset["instance"] == instance) & (subset["solver"] == solver)
                ]
                if row.empty:
                    continue
                value = float(row.iloc[0]["objective"])
                if not np.isfinite(value):
                    continue
                values.append(value)
                positions.append(
                    centers[index] - 0.4 + bar_width * (offset + 0.5)
                )
            axis.bar(
                positions,
                values,
                width=bar_width,
                label=solver,
                color=SOLVER_COLORS[solver],
            )
        axis.set_title(f"{formulation}-CFLP")
        axis.set_xticks(centers)
        axis.set_xticklabels(instances)
        axis.set_xlabel("Instance size")
        axis.set_ylabel("CFLP objective")
        axis.legend()
        axis.grid(axis="y", alpha=0.3)

    figure.suptitle("Figure 1. Objective comparison (original CFLP objective space)")
    figure.tight_layout()
    return _save(figure, output_dir, "figure1_objective_comparison.png")


def plot_true_gap(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 2 — Gurobi true optimum 대비 gap(%)."""
    instances = _ordered_instances(results)
    figure, axis = plt.subplots(figsize=(10, 4.5))

    markers = {"SS": "o", "MS": "s"}
    line_styles = {"SA": "-", "QA": "--"}

    for formulation in FORMULATIONS:
        for solver in ("SA", "QA"):
            subset = results[
                (results["formulation"] == formulation)
                & (results["solver"] == solver)
            ]
            xs, ys = [], []
            for index, instance in enumerate(instances):
                row = subset[subset["instance"] == instance]
                if row.empty:
                    continue
                value = float(row.iloc[0]["true_gap_percent"])
                if not np.isfinite(value):
                    continue
                xs.append(index)
                ys.append(value)
            if xs:
                axis.plot(
                    xs,
                    ys,
                    marker=markers[formulation],
                    linestyle=line_styles[solver],
                    color=SOLVER_COLORS[solver],
                    label=f"{formulation} / {solver}",
                )

    axis.axhline(0.0, color="#2c3e50", linewidth=1.2, label="Gurobi (0%)")
    axis.set_xticks(range(len(instances)))
    axis.set_xticklabels(instances)
    axis.set_xlabel("Instance size")
    axis.set_ylabel("True optimality gap (%)")
    axis.set_title("Figure 2. True optimality gap vs Gurobi optimum")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return _save(figure, output_dir, "figure2_true_gap.png")


def plot_runtime(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 3 — solver별 runtime 비교 (log scale).

    QA는 wall-clock과 QPU access time을 구분하여 표시한다.
    """
    instances = _ordered_instances(results)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for axis, formulation in zip(axes, FORMULATIONS):
        subset = results[results["formulation"] == formulation]
        series_names = list(SOLVER_ORDER) + ["QA (QPU only)"]
        centers, bar_width = _bar_positions(len(instances), len(series_names))
        for offset, name in enumerate(series_names):
            values, positions = [], []
            for index, instance in enumerate(instances):
                if name == "QA (QPU only)":
                    row = subset[
                        (subset["instance"] == instance) & (subset["solver"] == "QA")
                    ]
                    if row.empty or "qa_qpu_access_time_us" not in row:
                        continue
                    micro = float(row.iloc[0].get("qa_qpu_access_time_us", np.nan))
                    value = micro / 1e6 if np.isfinite(micro) else np.nan
                else:
                    row = subset[
                        (subset["instance"] == instance) & (subset["solver"] == name)
                    ]
                    if row.empty:
                        continue
                    value = float(row.iloc[0]["runtime"])
                if not np.isfinite(value) or value <= 0:
                    continue
                values.append(value)
                positions.append(centers[index] - 0.4 + bar_width * (offset + 0.5))
            color = SOLVER_COLORS.get(name, "#7f8c8d")
            axis.bar(positions, values, width=bar_width, label=name, color=color)
        axis.set_yscale("log")
        axis.set_title(f"{formulation}-CFLP")
        axis.set_xticks(centers)
        axis.set_xticklabels(instances)
        axis.set_xlabel("Instance size")
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Runtime (s, log scale)")
    axes[0].legend(fontsize=8)

    figure.suptitle("Figure 3. Runtime comparison")
    figure.tight_layout()
    return _save(figure, output_dir, "figure3_runtime.png")


def plot_qubo_size(qubo_stats: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 4 — instance 크기에 따른 QUBO 변수 수 증가."""
    instances = _ordered_instances(qubo_stats)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) 총 변수 수 추이
    axis = axes[0]
    for formulation in FORMULATIONS:
        subset = qubo_stats[qubo_stats["formulation"] == formulation].sort_values("size")
        axis.plot(
            subset["size"],
            subset["qubo_variables"],
            marker="o",
            color=FORMULATION_COLORS[formulation],
            label=f"{formulation} total",
        )
        axis.plot(
            subset["size"],
            subset["slack_variables"],
            marker="^",
            linestyle="--",
            color=FORMULATION_COLORS[formulation],
            alpha=0.6,
            label=f"{formulation} slack",
        )
    axis.set_xlabel("Instance size (|I| = |J|)")
    axis.set_ylabel("Number of binary variables")
    axis.set_title("(a) QUBO variable growth")
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    # (b) 변수 구성 분해 (stacked bar)
    axis = axes[1]
    centers, bar_width = _bar_positions(len(instances), len(FORMULATIONS))
    components = [
        ("decision_variables", "#34495e", "decision"),
        ("encoding_variables", "#f39c12", "encoding"),
        ("slack_variables", "#95a5a6", "slack"),
    ]
    for offset, formulation in enumerate(FORMULATIONS):
        subset = qubo_stats[qubo_stats["formulation"] == formulation]
        bottoms = np.zeros(len(instances))
        positions = centers - 0.4 + bar_width * (offset + 0.5)
        for column, color, label in components:
            values = np.array(
                [
                    float(
                        subset[subset["instance"] == instance][column].iloc[0]
                    )
                    if not subset[subset["instance"] == instance].empty
                    else 0.0
                    for instance in instances
                ]
            )
            axis.bar(
                positions,
                values,
                bottom=bottoms,
                width=bar_width,
                color=color,
                label=f"{formulation} {label}" if offset == 0 or True else None,
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms += values
        for position, total in zip(positions, bottoms):
            axis.text(
                position,
                total,
                formulation,
                ha="center",
                va="bottom",
                fontsize=7,
            )
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label.split(" ", 1)[-1], handle)
    axis.legend(unique.values(), unique.keys(), fontsize=8)
    axis.set_xticks(centers)
    axis.set_xticklabels(instances)
    axis.set_xlabel("Instance size")
    axis.set_ylabel("Number of binary variables")
    axis.set_title("(b) Variable composition")
    axis.grid(axis="y", alpha=0.3)

    figure.suptitle("Figure 4. QUBO size: SS vs MS")
    figure.tight_layout()
    return _save(figure, output_dir, "figure4_qubo_size.png")


def plot_feasibility(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 5 — SA/QA가 얻은 sample 중 feasible 비율."""
    instances = _ordered_instances(results)
    figure, axis = plt.subplots(figsize=(10, 4.5))

    series = [(f, s) for f in FORMULATIONS for s in ("SA", "QA")]
    centers, bar_width = _bar_positions(len(instances), len(series))
    hatches = {"SS": "", "MS": "//"}

    for offset, (formulation, solver) in enumerate(series):
        subset = results[
            (results["formulation"] == formulation) & (results["solver"] == solver)
        ]
        values, positions = [], []
        for index, instance in enumerate(instances):
            row = subset[subset["instance"] == instance]
            if row.empty:
                continue
            value = float(row.iloc[0]["feasible_fraction"])
            if not np.isfinite(value):
                continue
            values.append(value * 100.0)
            positions.append(centers[index] - 0.4 + bar_width * (offset + 0.5))
        axis.bar(
            positions,
            values,
            width=bar_width,
            color=SOLVER_COLORS[solver],
            hatch=hatches[formulation],
            edgecolor="white",
            label=f"{formulation} / {solver}",
        )

    axis.set_xticks(centers)
    axis.set_xticklabels(instances)
    axis.set_xlabel("Instance size")
    axis.set_ylabel("Feasible samples (%)")
    axis.set_title("Figure 5. Feasible solution rate among all reads")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return _save(figure, output_dir, "figure5_feasibility.png")


def plot_coefficient_range(
    qubo_stats: pd.DataFrame, output_dir: str | Path
) -> Path:
    """Figure 6 — QUBO 계수의 최소/최대 및 범위 (log scale)."""
    instances = _ordered_instances(qubo_stats)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axis = axes[0]
    for formulation in FORMULATIONS:
        subset = qubo_stats[qubo_stats["formulation"] == formulation].sort_values("size")
        xs = np.arange(len(subset))
        axis.plot(
            xs,
            subset["qubo_max"],
            marker="o",
            color=FORMULATION_COLORS[formulation],
            label=f"{formulation} max",
        )
        axis.plot(
            xs,
            np.abs(subset["qubo_min"]),
            marker="v",
            linestyle="--",
            color=FORMULATION_COLORS[formulation],
            alpha=0.7,
            label=f"{formulation} |min|",
        )
    axis.set_yscale("log")
    axis.set_xticks(range(len(instances)))
    axis.set_xticklabels(instances)
    axis.set_xlabel("Instance size")
    axis.set_ylabel("|QUBO coefficient| (log scale)")
    axis.set_title("(a) Extreme coefficients")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    axis = axes[1]
    centers, bar_width = _bar_positions(len(instances), len(FORMULATIONS))
    for offset, formulation in enumerate(FORMULATIONS):
        subset = qubo_stats[qubo_stats["formulation"] == formulation]
        values, positions = [], []
        for index, instance in enumerate(instances):
            row = subset[subset["instance"] == instance]
            if row.empty:
                continue
            values.append(float(row.iloc[0]["qubo_range"]))
            positions.append(centers[index] - 0.4 + bar_width * (offset + 0.5))
        axis.bar(
            positions,
            values,
            width=bar_width,
            color=FORMULATION_COLORS[formulation],
            label=formulation,
        )
    axis.set_yscale("log")
    axis.set_xticks(centers)
    axis.set_xticklabels(instances)
    axis.set_xlabel("Instance size")
    axis.set_ylabel("Coefficient range (max - min)")
    axis.set_title("(b) Dynamic range")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)

    figure.suptitle("Figure 6. QUBO coefficient range")
    figure.tight_layout()
    return _save(figure, output_dir, "figure6_coefficient_range.png")


def plot_embedding_feasibility(
    embedding_table: pd.DataFrame, output_dir: str | Path
) -> Path:
    """Figure 7 — instance 크기별 QA embedding 가능 여부."""
    instances = _ordered_instances(embedding_table)
    figure, axis = plt.subplots(figsize=(8, 3.4))

    status_color = {
        "OK": "#27ae60",
        "NOT_EMBEDDABLE": "#c0392b",
        "NO_QPU_ACCESS": "#7f8c8d",
        "NOT_ATTEMPTED": "#bdc3c7",
    }

    for row_index, formulation in enumerate(FORMULATIONS):
        subset = embedding_table[embedding_table["formulation"] == formulation]
        for column_index, instance in enumerate(instances):
            row = subset[subset["instance"] == instance]
            status = (
                str(row.iloc[0]["embedding_status"]) if not row.empty else "NOT_ATTEMPTED"
            )
            color = status_color.get(status, "#bdc3c7")
            axis.add_patch(
                plt.Rectangle(
                    (column_index - 0.45, row_index - 0.4),
                    0.9,
                    0.8,
                    color=color,
                    alpha=0.85,
                )
            )
            mark = "O" if status == "OK" else "X"
            label = mark if status in ("OK", "NOT_EMBEDDABLE") else status
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    axis.set_xlim(-0.6, len(instances) - 0.4)
    axis.set_ylim(-0.6, len(FORMULATIONS) - 0.4)
    axis.set_xticks(range(len(instances)))
    axis.set_xticklabels(instances)
    axis.set_yticks(range(len(FORMULATIONS)))
    axis.set_yticklabels(list(FORMULATIONS))
    axis.set_xlabel("Instance size")
    axis.set_title("Figure 7. QA embedding feasibility")
    axis.grid(False)
    figure.tight_layout()
    return _save(figure, output_dir, "figure7_embedding.png")


def plot_instance_overview(
    instances: Sequence[Any], output_dir: str | Path
) -> Path:
    """생성된 instance의 위치/수요/용량 분포를 시각화한다 (notebook 01용)."""
    num = len(instances)
    figure, axes = plt.subplots(2, num, figsize=(4 * num, 7))
    if num == 1:
        axes = axes.reshape(2, 1)

    for column, instance in enumerate(instances):
        axis = axes[0, column]
        axis.scatter(
            instance.customer_coords[:, 0],
            instance.customer_coords[:, 1],
            s=instance.demands * 3,
            color="#2980b9",
            alpha=0.7,
            label="customers",
        )
        axis.scatter(
            instance.facility_coords[:, 0],
            instance.facility_coords[:, 1],
            s=instance.capacities * 1.5,
            marker="s",
            facecolors="none",
            edgecolors="#c0392b",
            label="facilities",
        )
        axis.set_title(f"{instance.name} (ratio={instance.capacity_ratio:.2f})")
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.05, 1.05)
        if column == 0:
            axis.legend(fontsize=7)

        axis = axes[1, column]
        axis.scatter(
            instance.capacities,
            instance.fixed_costs,
            color="#8e44ad",
        )
        axis.set_xlabel("capacity s_j")
        if column == 0:
            axis.set_ylabel("fixed cost f_j")
        axis.set_title("capacity vs fixed cost")
        axis.grid(alpha=0.3)

    figure.suptitle("Generated CFLP instances")
    figure.tight_layout()
    return _save(figure, output_dir, "figure0_instances.png")


def generate_all_figures(
    results: pd.DataFrame,
    qubo_stats: pd.DataFrame,
    embedding_table: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Figure 1 ~ 7을 모두 생성한다."""
    return [
        plot_objective_comparison(results, output_dir),
        plot_true_gap(results, output_dir),
        plot_runtime(results, output_dir),
        plot_qubo_size(qubo_stats, output_dir),
        plot_feasibility(results, output_dir),
        plot_coefficient_range(qubo_stats, output_dir),
        plot_embedding_feasibility(embedding_table, output_dir),
    ]
