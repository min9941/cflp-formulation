"""Single-source CFLP (SS-CFLP) formulation 정의 및 평가.

Formulation
-----------
변수
    y_j in {0,1}          : 시설 j 개설 여부
    x_ij in {0,1}         : 고객 i를 시설 j에 배정하는지 여부

목적함수
    min  sum_j f_j y_j + sum_i sum_j c_ij d_i x_ij

제약
    (A) single assignment : sum_j x_ij = 1            for all i
    (C) capacity          : sum_i d_i x_ij <= s_j y_j for all j

Linking 제약(x_ij <= y_j)에 대하여
----------------------------------
d_i > 0, x_ij >= 0 이므로 (C)에서 y_j = 0이면 좌변이 0 이하가 되어
모든 x_ij = 0이 강제된다. 즉 linking 제약은 feasible set을 전혀
바꾸지 않는 **redundant 제약**이다. 본 실험에서는 SS/MS/Gurobi/QUBO
모두에서 linking 제약을 제거하여 동일한 feasible set을 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data_generator import CFLPInstance


@dataclass
class SSSolution:
    """SS-CFLP의 해.

    Attributes:
        x: (|I|, |J|) 배정 행렬. QUBO 해의 경우 0/1이지만 검증 목적상 실수 허용.
        y: 길이 |J|의 시설 개설 벡터.
    """

    x: np.ndarray
    y: np.ndarray


@dataclass
class FeasibilityReport:
    """원래 CFLP formulation 기준 feasibility 검증 결과."""

    is_feasible: bool
    max_violation: float
    total_violation: float
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "is_feasible": self.is_feasible,
            "max_violation": self.max_violation,
            "total_violation": self.total_violation,
        }
        payload.update(self.details)
        return payload


def evaluate_objective(instance: CFLPInstance, solution: SSSolution) -> float:
    """SS-CFLP의 원래 목적함수 값을 계산한다.

    QUBO energy가 아니라 **원래 CFLP objective space**에서 계산한다.
    """
    fixed_part = float(np.dot(instance.fixed_costs, solution.y))
    weighted_costs = instance.transport_costs * instance.demands[:, None]
    transport_part = float((weighted_costs * solution.x).sum())
    return fixed_part + transport_part


def check_feasibility(
    instance: CFLPInstance, solution: SSSolution, tolerance: float = 1e-6
) -> FeasibilityReport:
    """SS 해를 원래 formulation 기준으로 검증한다.

    검증 항목:
        - assignment violation : |sum_j x_ij - 1|
        - capacity violation   : max(0, sum_i d_i x_ij - s_j y_j)
        - linking violation    : max(0, x_ij - y_j)  (진단용 기록)
          linking은 formulation에서 제거했으므로 feasibility 판정에는
          사용하지 않고, capacity violation에 이미 포함되어 있다.
        - binary violation     : x, y가 0/1에서 벗어난 정도
    """
    x = np.asarray(solution.x, dtype=float)
    y = np.asarray(solution.y, dtype=float)

    assignment_residual = np.abs(x.sum(axis=1) - 1.0)
    capacity_residual = np.maximum(
        0.0, (instance.demands[:, None] * x).sum(axis=0) - instance.capacities * y
    )
    linking_residual = np.maximum(0.0, x - y[None, :])
    binary_residual = np.concatenate(
        [np.abs(x.ravel() - np.rint(x.ravel())), np.abs(y - np.rint(y))]
    )

    # feasibility 판정에 사용하는 제약: assignment + capacity
    binding = np.concatenate([assignment_residual, capacity_residual])
    max_violation = float(binding.max()) if binding.size else 0.0
    total_violation = float(binding.sum())

    is_feasible = bool(
        max_violation <= tolerance and float(binary_residual.max(initial=0.0)) <= tolerance
    )

    return FeasibilityReport(
        is_feasible=is_feasible,
        max_violation=max_violation,
        total_violation=total_violation,
        details={
            "assignment_max_violation": float(assignment_residual.max(initial=0.0)),
            "assignment_total_violation": float(assignment_residual.sum()),
            "capacity_max_violation": float(capacity_residual.max(initial=0.0)),
            "capacity_total_violation": float(capacity_residual.sum()),
            "linking_max_violation": float(linking_residual.max(initial=0.0)),
            "linking_total_violation": float(linking_residual.sum()),
            "binary_max_violation": float(binary_residual.max(initial=0.0)),
        },
    )


def objective_upper_bound(instance: CFLPInstance) -> float:
    """목적함수로부터 얻을 수 있는 최대 이득의 보수적 상한.

    SS 목적함수의 모든 계수(f_j, c_ij d_i)가 비음수이므로, 어떤 변수
    조합을 0으로 만들더라도 줄일 수 있는 목적값의 상한은 전체 계수 합이다.
    이 값이 penalty coefficient 계산의 기준이 된다.
    """
    weighted_costs = instance.transport_costs * instance.demands[:, None]
    return float(instance.fixed_costs.sum() + weighted_costs.sum())
