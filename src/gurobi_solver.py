"""Gurobi를 이용한 원래 MILP 해결 모듈.

Gurobi 결과는 SA/QA 비교의 **true optimum reference**로 사용한다.

세 가지 reference를 계산한다.
    1. SS        : x_ij in {0,1}
    2. MS        : x_ij in [0,1]  (원래 연속 formulation)
    3. MS-int-q  : q_ij in {0,...,d_i} 정수 (QUBO와 동일한 이산화 수준)

MS와 MS-int-q를 모두 기록함으로써, QUBO 기반 해의 gap에서
"1 unit 이산화로 인한 손실"과 "solver 품질로 인한 손실"을 분리할 수 있다.

Linking 제약(x_ij <= y_j)은 capacity 제약에 의해 함의되는 redundant
제약이므로 제거한다 (SS/MS/QUBO 모두 동일한 feasible set 사용).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .cflp_ms import MSSolution
from .cflp_ss import SSSolution
from .data_generator import CFLPInstance

_STATUS_NAMES: dict[int, str] = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
}


@dataclass
class GurobiResult:
    """Gurobi 해결 결과."""

    model_name: str
    objective: float
    status: str
    runtime: float
    mip_gap: float
    num_variables: int
    num_constraints: int
    solution: SSSolution | MSSolution
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """결과 표 저장용 dict."""
        record = {
            "gurobi_model": self.model_name,
            "objective": self.objective,
            "status": self.status,
            "runtime": self.runtime,
            "mip_gap": self.mip_gap,
            "milp_variables": self.num_variables,
            "milp_constraints": self.num_constraints,
        }
        record.update(self.extra)
        return record


def _status_name(status_code: int) -> str:
    return _STATUS_NAMES.get(status_code, f"STATUS_{status_code}")


def _apply_parameters(model: gp.Model, gurobi_config: dict[str, Any]) -> None:
    """설정 파일의 Gurobi 파라미터를 적용한다."""
    model.Params.OutputFlag = int(gurobi_config.get("output_flag", 0))
    model.Params.MIPGap = float(gurobi_config.get("mip_gap", 0.0))
    model.Params.TimeLimit = float(gurobi_config.get("time_limit", 600))
    model.Params.Seed = int(gurobi_config.get("seed", 0))


def solve_ss(
    instance: CFLPInstance, gurobi_config: dict[str, Any]
) -> GurobiResult:
    """SS-CFLP를 Gurobi로 해결한다.

    min  sum_j f_j y_j + sum_ij c_ij d_i x_ij
    s.t. sum_j x_ij = 1                  (모든 i)
         sum_i d_i x_ij <= s_j y_j       (모든 j)
         x_ij, y_j in {0,1}
    """
    num_customers = instance.num_customers
    num_facilities = instance.num_facilities

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("SS-CFLP", env=env) as model:
        _apply_parameters(model, gurobi_config)

        x = model.addVars(num_customers, num_facilities, vtype=GRB.BINARY, name="x")
        y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")

        model.setObjective(
            gp.quicksum(
                float(instance.fixed_costs[j]) * y[j] for j in range(num_facilities)
            )
            + gp.quicksum(
                float(instance.transport_costs[i, j]) * float(instance.demands[i])
                * x[i, j]
                for i in range(num_customers)
                for j in range(num_facilities)
            ),
            GRB.MINIMIZE,
        )

        for i in range(num_customers):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(num_facilities)) == 1,
                name=f"assign_{i}",
            )
        for j in range(num_facilities):
            model.addConstr(
                gp.quicksum(
                    float(instance.demands[i]) * x[i, j]
                    for i in range(num_customers)
                )
                <= float(instance.capacities[j]) * y[j],
                name=f"capacity_{j}",
            )

        model.update()
        num_variables = model.NumVars
        num_constraints = model.NumConstrs

        start = time.perf_counter()
        model.optimize()
        wall_clock = time.perf_counter() - start

        status = _status_name(model.Status)
        if model.SolCount == 0:
            raise RuntimeError(
                f"SS instance '{instance.name}'에서 Gurobi가 실행가능해를 찾지 "
                f"못했습니다 (status={status})."
            )

        x_values = np.array(
            [
                [x[i, j].X for j in range(num_facilities)]
                for i in range(num_customers)
            ],
            dtype=float,
        )
        y_values = np.array([y[j].X for j in range(num_facilities)], dtype=float)
        objective = float(model.ObjVal)
        mip_gap = float(model.MIPGap)

    return GurobiResult(
        model_name="SS",
        objective=objective,
        status=status,
        runtime=wall_clock,
        mip_gap=mip_gap,
        num_variables=num_variables,
        num_constraints=num_constraints,
        solution=SSSolution(x=np.rint(x_values), y=np.rint(y_values)),
    )


def _solve_ms_core(
    instance: CFLPInstance,
    gurobi_config: dict[str, Any],
    integral: bool,
) -> GurobiResult:
    """MS-CFLP를 unit 변수 q로 해결한다.

    Args:
        integral: True이면 q_ij를 정수로 제한한다(QUBO와 동일한 이산화).
            False이면 q_ij를 연속으로 두며, 이는 원래 MS formulation
            (x_ij in [0,1])과 정확히 동일하다.
    """
    num_customers = instance.num_customers
    num_facilities = instance.num_facilities
    vtype = GRB.INTEGER if integral else GRB.CONTINUOUS
    model_name = "MS-int-q" if integral else "MS"

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(model_name, env=env) as model:
        _apply_parameters(model, gurobi_config)

        q = model.addVars(
            num_customers,
            num_facilities,
            lb=0.0,
            ub={
                (i, j): float(instance.demands[i])
                for i in range(num_customers)
                for j in range(num_facilities)
            },
            vtype=vtype,
            name="q",
        )
        y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")

        model.setObjective(
            gp.quicksum(
                float(instance.fixed_costs[j]) * y[j] for j in range(num_facilities)
            )
            + gp.quicksum(
                float(instance.transport_costs[i, j]) * q[i, j]
                for i in range(num_customers)
                for j in range(num_facilities)
            ),
            GRB.MINIMIZE,
        )

        for i in range(num_customers):
            model.addConstr(
                gp.quicksum(q[i, j] for j in range(num_facilities))
                == float(instance.demands[i]),
                name=f"demand_{i}",
            )
        for j in range(num_facilities):
            model.addConstr(
                gp.quicksum(q[i, j] for i in range(num_customers))
                <= float(instance.capacities[j]) * y[j],
                name=f"capacity_{j}",
            )

        model.update()
        num_variables = model.NumVars
        num_constraints = model.NumConstrs

        start = time.perf_counter()
        model.optimize()
        wall_clock = time.perf_counter() - start

        status = _status_name(model.Status)
        if model.SolCount == 0:
            raise RuntimeError(
                f"MS instance '{instance.name}'에서 Gurobi가 실행가능해를 찾지 "
                f"못했습니다 (status={status}, integral={integral})."
            )

        q_values = np.array(
            [
                [q[i, j].X for j in range(num_facilities)]
                for i in range(num_customers)
            ],
            dtype=float,
        )
        y_values = np.array([y[j].X for j in range(num_facilities)], dtype=float)
        objective = float(model.ObjVal)
        mip_gap = float(model.MIPGap)

    if integral:
        q_values = np.rint(q_values)

    return GurobiResult(
        model_name=model_name,
        objective=objective,
        status=status,
        runtime=wall_clock,
        mip_gap=mip_gap,
        num_variables=num_variables,
        num_constraints=num_constraints,
        solution=MSSolution(q=q_values, y=np.rint(y_values)),
    )


def solve_ms_continuous(
    instance: CFLPInstance, gurobi_config: dict[str, Any]
) -> GurobiResult:
    """원래 MS formulation(연속 allocation)을 해결한다."""
    return _solve_ms_core(instance, gurobi_config, integral=False)


def solve_ms_integer(
    instance: CFLPInstance, gurobi_config: dict[str, Any]
) -> GurobiResult:
    """QUBO와 동일한 1 unit 이산화를 적용한 MS를 해결한다."""
    return _solve_ms_core(instance, gurobi_config, integral=True)


def solve_all(
    instance: CFLPInstance, gurobi_config: dict[str, Any]
) -> dict[str, GurobiResult]:
    """하나의 instance에 대한 세 가지 reference를 모두 계산한다."""
    return {
        "SS": solve_ss(instance, gurobi_config),
        "MS": solve_ms_continuous(instance, gurobi_config),
        "MS_INT": solve_ms_integer(instance, gurobi_config),
    }
