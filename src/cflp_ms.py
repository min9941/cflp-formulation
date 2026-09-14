"""Multiple-source CFLP (MS-CFLP) formulation 정의 및 평가.

Formulation (원래 형태)
-----------------------
변수
    y_j in {0,1}          : 시설 j 개설 여부
    x_ij in [0,1]         : 고객 i의 수요 중 시설 j가 담당하는 비율

목적함수
    min  sum_j f_j y_j + sum_i sum_j c_ij d_i x_ij

제약
    (D) demand   : sum_j x_ij = 1                 for all i
    (C) capacity : sum_i d_i x_ij <= s_j y_j      for all j

Unit 표현 (1 unit shipment precision)
-------------------------------------
    q_ij = d_i x_ij  로 두면  q_ij in [0, d_i],  x_ij = q_ij / d_i.

    목적함수 : min sum_j f_j y_j + sum_i sum_j c_ij q_ij
    (D)      : sum_j q_ij = d_i
    (C)      : sum_i q_ij <= s_j y_j

본 모듈의 ``MSSolution``은 항상 **unit 표현 q**를 저장한다.
연속 완화(Gurobi continuous)는 q가 실수, 정수 참조해(Gurobi integer-q)와
QUBO 해는 q가 정수라는 점만 다르며, 목적함수 계산식은 동일하다.

Linking 제약(x_ij <= y_j)은 SS와 같은 이유로 redundant하므로 제거한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cflp_ss import FeasibilityReport
from .data_generator import CFLPInstance


@dataclass
class MSSolution:
    """MS-CFLP의 해 (unit 표현).

    Attributes:
        q: (|I|, |J|) 배송량 행렬. q_ij = d_i * x_ij.
        y: 길이 |J|의 시설 개설 벡터.
    """

    q: np.ndarray
    y: np.ndarray

    def to_fractions(self, instance: CFLPInstance) -> np.ndarray:
        """원래 formulation의 비율 변수 x_ij = q_ij / d_i 로 복원한다."""
        return np.asarray(self.q, dtype=float) / instance.demands[:, None]


def evaluate_objective(instance: CFLPInstance, solution: MSSolution) -> float:
    """MS-CFLP의 원래 목적함수 값을 계산한다.

    sum_j f_j y_j + sum_ij c_ij d_i x_ij = sum_j f_j y_j + sum_ij c_ij q_ij
    """
    fixed_part = float(np.dot(instance.fixed_costs, solution.y))
    transport_part = float((instance.transport_costs * solution.q).sum())
    return fixed_part + transport_part


def check_feasibility(
    instance: CFLPInstance,
    solution: MSSolution,
    tolerance: float = 1e-6,
    require_integral: bool = False,
) -> FeasibilityReport:
    """MS 해를 원래 formulation 기준으로 검증한다.

    Args:
        instance: 대상 instance.
        solution: unit 표현 해.
        tolerance: feasibility 판정 허용오차.
        require_integral: True이면 q가 정수인지도 함께 검사한다
            (QUBO/SA/QA 해 검증용). Gurobi 연속해 검증 시에는 False.
    """
    q = np.asarray(solution.q, dtype=float)
    y = np.asarray(solution.y, dtype=float)

    demand_residual = np.abs(q.sum(axis=1) - instance.demands.astype(float))
    capacity_residual = np.maximum(
        0.0, q.sum(axis=0) - instance.capacities * y
    )
    # q >= 0 은 encoding상 항상 만족하지만 일반 검증을 위해 포함한다.
    nonneg_residual = np.maximum(0.0, -q)
    upper_residual = np.maximum(0.0, q - instance.demands[:, None])
    linking_residual = np.maximum(0.0, q - instance.demands[:, None] * y[None, :])

    binding = np.concatenate(
        [demand_residual, capacity_residual, nonneg_residual.ravel(), upper_residual.ravel()]
    )
    max_violation = float(binding.max()) if binding.size else 0.0
    total_violation = float(binding.sum())

    is_feasible = bool(max_violation <= tolerance)
    if require_integral:
        integral_residual = float(np.abs(q - np.rint(q)).max(initial=0.0))
        is_feasible = is_feasible and integral_residual <= tolerance
    binary_residual = float(np.abs(y - np.rint(y)).max(initial=0.0))
    is_feasible = is_feasible and binary_residual <= tolerance

    return FeasibilityReport(
        is_feasible=is_feasible,
        max_violation=max_violation,
        total_violation=total_violation,
        details={
            "demand_max_violation": float(demand_residual.max(initial=0.0)),
            "demand_total_violation": float(demand_residual.sum()),
            "capacity_max_violation": float(capacity_residual.max(initial=0.0)),
            "capacity_total_violation": float(capacity_residual.sum()),
            "linking_max_violation": float(linking_residual.max(initial=0.0)),
            "linking_total_violation": float(linking_residual.sum()),
            "binary_max_violation": binary_residual,
        },
    )


def objective_upper_bound(instance: CFLPInstance) -> float:
    """목적함수로부터 얻을 수 있는 최대 이득의 보수적 상한.

    MS QUBO에서 q_ij의 binary expansion 계수 합이 정확히 d_i이므로,
    운송비 항의 계수 총합은 sum_ij c_ij d_i가 되어 SS와 동일한 상한식을
    사용할 수 있다. 이로써 SS와 MS의 penalty 계산 기준이 일관된다.
    """
    weighted_costs = instance.transport_costs * instance.demands[:, None]
    return float(instance.fixed_costs.sum() + weighted_costs.sum())


def summarize_solution(instance: CFLPInstance, solution: MSSolution) -> dict[str, Any]:
    """해의 요약 통계 (notebook 출력용)."""
    q = np.asarray(solution.q, dtype=float)
    served_facilities = int((q.sum(axis=0) > 1e-9).sum())
    split_customers = int(((q > 1e-9).sum(axis=1) > 1).sum())
    return {
        "opened_facilities": int(np.rint(solution.y).sum()),
        "used_facilities": served_facilities,
        "split_customers": split_customers,
    }
