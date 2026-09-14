"""Embedding 실패 원인 분석 결과 시각화 (Figure 16 ~ 19).

축 레이블과 범례는 한글 폰트 의존성을 피하기 위해 영어로 작성한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TOPOLOGY_STYLE: dict[str, dict] = {
    "pegasus": {"color": "#2980b9", "marker": "o", "linestyle": "-"},
    "zephyr": {"color": "#27ae60", "marker": "s", "linestyle": "--"},
}
QUALITY_METRICS: tuple[tuple[str, str], ...] = (
    ("physical_qubits", "Physical qubits"),
    ("avg_chain_length", "Average chain length"),
    ("max_chain_length", "Maximum chain length"),
)


def _save(figure: plt.Figure, output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _style(topology: str) -> dict:
    return TOPOLOGY_STYLE.get(topology, {"color": "#7f8c8d", "marker": "^"})


def _instances(results: pd.DataFrame) -> list[str]:
    """논리 변수 수 기준으로 정렬된 instance 목록."""
    pairs = results[["instance", "logical_variables"]].drop_duplicates()
    return pairs.sort_values("logical_variables")["instance"].tolist()


def plot_timeout_success(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 16 — timeout 대 embedding 성공률 (instance별 panel)."""
    subset = results[results["sweep"] == "timeout"]
    instances = _instances(subset)
    if not instances:
        raise ValueError("timeout sweep 결과가 없습니다.")

    columns = len(instances)
    figure, axes = plt.subplots(
        1, columns, figsize=(4.0 * columns, 4.2), sharey=True, squeeze=False
    )
    for axis, instance in zip(axes[0], instances):
        part = subset[subset["instance"] == instance]
        for topology, group in part.groupby("topology"):
            rate = group.groupby("timeout")["success"].mean() * 100.0
            axis.plot(rate.index, rate.values, label=topology, **_style(topology))
        variables = int(part["logical_variables"].iloc[0])
        edges = int(part["logical_edges"].iloc[0])
        axis.set_title(f"{instance}\n{variables} vars, {edges} edges", fontsize=10)
        axis.set_xscale("log")
        axis.set_xlabel("Embedding timeout (s)")
        axis.set_ylim(-5, 105)
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("Embedding success rate (%)")
    axes[0][0].legend(fontsize=8)
    figure.suptitle("Figure 16. Embedding success rate vs search timeout")
    figure.tight_layout()
    return _save(figure, output_dir, "figure16_embedding_timeout.png")


def plot_embedding_quality(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 17 — 성공한 embedding의 품질 비교."""
    successes = results[results["success"]]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), squeeze=False)
    if successes.empty:
        for axis, (_, label) in zip(axes[0], QUALITY_METRICS):
            axis.text(
                0.5, 0.5, "no successful embedding",
                transform=axis.transAxes, ha="center", va="center",
            )
            axis.set_title(label, fontsize=10)
        figure.suptitle("Figure 17. Embedding quality (successful runs only)")
        figure.tight_layout()
        return _save(figure, output_dir, "figure17_embedding_quality.png")

    instances = _instances(successes)
    positions = np.arange(len(instances), dtype=float)
    for axis, (column, label) in zip(axes[0], QUALITY_METRICS):
        for offset, (topology, group) in enumerate(successes.groupby("topology")):
            values, xs = [], []
            for index, instance in enumerate(instances):
                part = group[group["instance"] == instance]
                if part.empty:
                    continue
                values.append(float(part[column].median()))
                xs.append(positions[index] - 0.2 + 0.4 * offset)
            axis.bar(
                xs, values, width=0.38, label=topology,
                color=_style(topology)["color"],
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(instances)
        axis.set_ylabel(label)
        axis.set_title(label, fontsize=10)
        axis.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    figure.suptitle("Figure 17. Embedding quality (median over successful runs)")
    figure.tight_layout()
    return _save(figure, output_dir, "figure17_embedding_quality.png")


def plot_tries_success(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 18 — tries 대 embedding 성공률 (timeout 고정)."""
    subset = results[results["sweep"] == "tries"]
    if subset.empty:
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        axis.text(
            0.5, 0.5, "no tries sweep result",
            transform=axis.transAxes, ha="center", va="center",
        )
        axis.set_title("Figure 18. Embedding success rate vs tries")
        return _save(figure, output_dir, "figure18_embedding_tries.png")

    instances = _instances(subset)
    columns = len(instances)
    figure, axes = plt.subplots(
        1, columns, figsize=(4.0 * columns, 4.2), sharey=True, squeeze=False
    )
    fixed_timeout = int(subset["timeout"].iloc[0])
    for axis, instance in zip(axes[0], instances):
        part = subset[subset["instance"] == instance]
        for topology, group in part.groupby("topology"):
            rate = group.groupby("tries")["success"].mean() * 100.0
            axis.plot(rate.index, rate.values, label=topology, **_style(topology))
        axis.set_title(f"{instance}", fontsize=10)
        axis.set_xlabel("tries")
        axis.set_ylim(-5, 105)
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("Embedding success rate (%)")
    axes[0][0].legend(fontsize=8)
    figure.suptitle(
        f"Figure 18. Embedding success rate vs tries (timeout = {fixed_timeout}s)"
    )
    figure.tight_layout()
    return _save(figure, output_dir, "figure18_embedding_tries.png")


def plot_density_vs_success(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Figure 19 — QUBO 그래프 규모와 embedding 성공률의 관계.

    변수 수가 아니라 **엣지 수**가 성공 여부를 가르는지 확인하는 그림이다.
    """
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for topology, group in results.groupby("topology"):
        aggregated = group.groupby("instance").agg(
            edges=("logical_edges", "first"),
            variables=("logical_variables", "first"),
            rate=("success", "mean"),
        )
        axis.scatter(
            aggregated["edges"], aggregated["rate"] * 100.0,
            s=aggregated["variables"] / 3.0 + 30,
            label=topology, color=_style(topology)["color"], alpha=0.75,
        )
        for name, row in aggregated.iterrows():
            axis.annotate(
                name, (row["edges"], row["rate"] * 100.0),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
    axis.set_xscale("log")
    axis.set_xlabel("QUBO quadratic terms (log scale)   —  marker size = variables")
    axis.set_ylabel("Embedding success rate (%)")
    axis.set_ylim(-5, 105)
    axis.set_title("Figure 19. Graph size vs embedding success")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, output_dir, "figure19_density_vs_success.png")


def generate_all(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Figure 16 ~ 19를 생성한다. timeout sweep이 없으면 해당 그림은 건너뛴다."""
    paths: list[Path] = []
    if (results["sweep"] == "timeout").any():
        paths.append(plot_timeout_success(results, output_dir))
    paths.append(plot_embedding_quality(results, output_dir))
    paths.append(plot_tries_success(results, output_dir))
    paths.append(plot_density_vs_success(results, output_dir))
    return paths
