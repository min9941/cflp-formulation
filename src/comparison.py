"""linking constraint 포함/제외 비교용 헬퍼.

notebook 04, 05가 얇게 유지되도록 비교 로직을 여기에 둔다.

비교 대상
--------
1. ``linking_effect``    : 같은 formulation 안에서 linking 포함 vs 제외
2. ``formulation_gap``   : 같은 linking 설정 안에서 SS vs MS

linking constraint(x_ij <= y_j)는 capacity constraint에 의해 함의되는
redundant 제약이다. 따라서 feasible region이 동일하고, 관측되는 모든 차이는
**QUBO 표현에 드는 비용**이다. 이것이 이 비교의 요점이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SIZE_COLUMNS = ("qubo_variables", "qubo_quadratic_terms")


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """(instance, formulation) x linking 형태로 펼친다."""
    return frame.pivot_table(
        index=["instance", "size", "formulation"],
        columns="linking",
        values=column,
        dropna=False,
    )


def linking_effect(results: pd.DataFrame) -> pd.DataFrame:
    """linking 포함이 QUBO 크기와 solver 성능에 미친 영향.

    Args:
        results: ``linking`` 컬럼을 갖는 SA 또는 QA 결과 표.

    Returns:
        instance x formulation 별 비교표. ``ratio`` 컬럼이 1보다 크면
        linking을 넣었을 때 그만큼 커졌다는 뜻이다.
    """
    rows: list[dict] = []
    for (instance, size, formulation), group in results.groupby(
        ["instance", "size", "formulation"], sort=False
    ):
        without = group[~group["linking"].astype(bool)]
        with_link = group[group["linking"].astype(bool)]
        if without.empty or with_link.empty:
            continue
        row: dict = {
            "instance": instance,
            "size": size,
            "formulation": formulation,
        }
        for column, label in zip(_SIZE_COLUMNS, ("vars", "terms")):
            if column not in group:
                continue
            a = float(without[column].iloc[0])
            b = float(with_link[column].iloc[0])
            row[f"{label}_without"] = int(a)
            row[f"{label}_with"] = int(b)
            row[f"{label}_ratio"] = round(b / a, 2) if a else np.nan
        for column, label in (
            ("feasible_fraction", "feasible"),
            ("best_feasible_objective", "objective"),
            ("runtime", "runtime"),
        ):
            if column not in group:
                continue
            row[f"{label}_without"] = without[column].iloc[0]
            row[f"{label}_with"] = with_link[column].iloc[0]
        if "status" in group:
            row["status_without"] = without["status"].iloc[0]
            row["status_with"] = with_link["status"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["formulation", "size"])


def formulation_gap(results: pd.DataFrame, linking: bool) -> pd.DataFrame:
    """같은 linking 설정 안에서 SS와 MS를 비교한다.

    Args:
        results: ``linking`` 컬럼을 갖는 결과 표.
        linking: True면 linking 포함끼리, False면 제외끼리 비교한다.
    """
    subset = results[results["linking"].astype(bool) == bool(linking)]
    rows: list[dict] = []
    for (instance, size), group in subset.groupby(["instance", "size"], sort=False):
        single = group[group["formulation"] == "SS"]
        multiple = group[group["formulation"] == "MS"]
        if single.empty or multiple.empty:
            continue
        row: dict = {"instance": instance, "size": size, "linking": bool(linking)}
        for column, label in zip(_SIZE_COLUMNS, ("vars", "terms")):
            if column not in group:
                continue
            a = float(single[column].iloc[0])
            b = float(multiple[column].iloc[0])
            row[f"{label}_SS"] = int(a)
            row[f"{label}_MS"] = int(b)
            row[f"{label}_MS_over_SS"] = round(b / a, 2) if a else np.nan
        for column, label in (
            ("feasible_fraction", "feasible"),
            ("best_feasible_objective", "objective"),
        ):
            if column not in group:
                continue
            row[f"{label}_SS"] = single[column].iloc[0]
            row[f"{label}_MS"] = multiple[column].iloc[0]
        if "status" in group:
            row["status_SS"] = single["status"].iloc[0]
            row["status_MS"] = multiple["status"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("size")


def embedding_summary(results: pd.DataFrame) -> pd.DataFrame:
    """QA 결과에서 linking 유무별 embedding 성공 여부를 정리한다."""
    columns = [
        column
        for column in (
            "instance", "size", "formulation", "linking",
            "qubo_variables", "qubo_quadratic_terms",
            "status", "physical_qubits", "max_chain_length",
        )
        if column in results.columns
    ]
    return results[columns].sort_values(["formulation", "size", "linking"])
