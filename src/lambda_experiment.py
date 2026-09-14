"""Penalty coefficient(lambda) 민감도 실험 모듈.

목적
----
본 실험(notebook 04, 05)에서는 lambda를 실험 시작 전에 고정하고 결과를 보고
조정하지 않았다. 그 결과 QA는 24회 중 1회만 feasible solution을 냈고,
계수 범위가 10^9에 달했다.

이 모듈은 **왜 그런지를 진단하기 위한 별도 실험**이다. lambda를 의도적으로
넓은 범위에서 변화시키며 다음 두 가설을 구분한다.

    가설 A (precision) : lambda가 커서 계수 범위가 넓어지고, objective 정보가
                         하드웨어의 아날로그 정밀도 아래로 묻힌다.
    가설 B (landscape) : lambda가 커서 penalty 지형이 험해지고, solver가
                         지형 자체를 탐색하지 못한다.

구분 방법
---------
SA와 QA를 **동일한 QUBO**에 대해 같은 lambda grid로 실행한다.

    - SA는 소프트웨어이므로 아날로그 정밀도 제약이 없다.
    - 따라서 lambda를 낮췄을 때 QA만 개선되면 가설 A,
      SA와 QA가 함께 개선되면 가설 B의 기여가 크다는 뜻이다.

lambda 기준점
-------------
lambda의 이론적 하한은 U_obj가 아니라 **Z_ub(알려진 feasible solution의
objective)** 이다. infeasible solution의 energy는 최소 lambda인 반면,
optimal solution의 energy는 Z* <= Z_ub이므로 lambda > Z_ub이면 ground state가
feasible임이 보장된다.

따라서 grid를 m = lambda / Z_ub 로 두고, m = 1이 이론적 경계가 되도록 한다.
m < 1 구간은 이론적 보장이 깨지는 영역이며, 진단 목적으로만 사용하고
본 실험 결과로는 보고하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import sa_solver
from .data_generator import CFLPInstance
from .evaluation import SolverOutcome, analyze_sampleset, failed_outcome
from .qubo_builder import QUBOModel, build_qubo

# 이론적 경계(m = 1)를 반드시 포함하고, 그 위아래를 로그 간격으로 훑는다.
DEFAULT_MULTIPLIERS: tuple[float, ...] = (
    0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0
)


@dataclass
class LambdaPoint:
    """lambda grid 한 점에서의 관측 결과."""

    multiplier: float
    lambda_value: float
    qubo_min: float
    qubo_max: float
    qubo_range: float
    max_abs_coupling: float
    scaled_objective_coefficient: float
    solver: str
    status: str
    feasible_fraction: float
    best_feasible_objective: float | None
    true_gap_percent: float
    ground_state_is_feasible: bool
    runtime: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = {
            "multiplier": self.multiplier,
            "lambda_value": self.lambda_value,
            "qubo_min": self.qubo_min,
            "qubo_max": self.qubo_max,
            "qubo_range": self.qubo_range,
            "max_abs_coupling": self.max_abs_coupling,
            "scaled_objective_coefficient": self.scaled_objective_coefficient,
            "solver": self.solver,
            "status": self.status,
            "feasible_fraction": self.feasible_fraction,
            "best_feasible_objective": self.best_feasible_objective,
            "true_gap_percent": self.true_gap_percent,
            "ground_state_is_feasible": self.ground_state_is_feasible,
            "runtime": self.runtime,
        }
        record.update(self.extra)
        return record


def lambda_grid(
    reference_objective: float, multipliers: Sequence[float] = DEFAULT_MULTIPLIERS
) -> list[tuple[float, float]]:
    """(multiplier, lambda) 목록을 만든다.

    Args:
        reference_objective: 기준값 Z_ub. 보통 Gurobi optimum을 쓴다.
        multipliers: Z_ub에 곱할 배수들.
    """
    return [(float(m), float(m) * float(reference_objective)) for m in multipliers]


def scaled_objective_coefficient(
    model: QUBOModel, instance: CFLPInstance
) -> tuple[float, float]:
    """auto_scale 이후 objective 계수가 얼마나 작아지는지 계산한다.

    D-Wave는 모든 계수를 max|J|로 나누어 하드웨어 범위에 맞춘다. 그 결과
    원래 objective 계수가 얼마나 되는지를 보면, ICE 노이즈(정규화 단위로
    대략 0.01~0.03) 대비 어느 위치인지 직접 비교할 수 있다.

    Returns:
        (max|J|, 정규화된 최소 objective 계수).
    """
    bqm = sa_solver.to_bqm(model)
    couplings = [abs(value) for value in bqm.spin.quadratic.values()]
    max_abs = max(couplings) if couplings else 1.0
    weighted = instance.transport_costs * instance.demands[:, None]
    smallest = float(min(weighted.min(), instance.fixed_costs.min()))
    return float(max_abs), float(smallest / max_abs)


def check_ground_state_feasible(
    model: QUBOModel,
    instance: CFLPInstance,
    reference_sample: np.ndarray,
    outcome: SolverOutcome,
) -> bool:
    """penalty가 충분한지 직접 확인한다.

    알려진 optimal solution을 encoding한 sample의 energy보다 더 낮은 energy를
    갖는 infeasible sample이 관측되면, 그 lambda에서는 ground state가
    feasible이라는 보장이 깨진 것이다.

    Returns:
        보장이 유지되면 True.
    """
    if outcome.best_sample is None:
        return True
    reference_energy = model.energy(reference_sample)
    if outcome.best_is_feasible:
        return True
    return bool(outcome.best_energy >= reference_energy - 1e-6)


def run_sa_sweep(
    instance: CFLPInstance,
    formulation: str,
    reference_objective: float,
    reference_sample: np.ndarray,
    sa_config: dict[str, Any],
    tolerance: float,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    precision: int = 1,
) -> list[LambdaPoint]:
    """lambda grid 전체에 대해 SA를 실행한다 (classical control)."""
    points: list[LambdaPoint] = []
    for multiplier, lam in lambda_grid(reference_objective, multipliers):
        model = build_qubo(instance, formulation, 1.1, precision, lambda_value=lam)
        max_abs, scaled = scaled_objective_coefficient(model, instance)
        outcome = sa_solver.solve(model, instance, sa_config, tolerance)
        stats = model.coefficient_stats()
        gap = (
            (outcome.best_feasible_objective - reference_objective)
            / abs(reference_objective) * 100.0
            if outcome.best_feasible_objective is not None
            else float("nan")
        )
        points.append(
            LambdaPoint(
                multiplier=multiplier,
                lambda_value=lam,
                qubo_min=stats["qubo_min"],
                qubo_max=stats["qubo_max"],
                qubo_range=stats["qubo_range"],
                max_abs_coupling=max_abs,
                scaled_objective_coefficient=scaled,
                solver="SA",
                status=outcome.status,
                feasible_fraction=outcome.feasible_fraction,
                best_feasible_objective=outcome.best_feasible_objective,
                true_gap_percent=gap,
                ground_state_is_feasible=check_ground_state_feasible(
                    model, instance, reference_sample, outcome
                ),
                runtime=outcome.runtime,
            )
        )
    return points


def run_qa_sweep(
    instance: CFLPInstance,
    formulation: str,
    reference_objective: float,
    reference_sample: np.ndarray,
    qa_config: dict[str, Any],
    embedding_config: dict[str, Any],
    tolerance: float,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    precision: int = 1,
) -> list[LambdaPoint]:
    """lambda grid 전체에 대해 QA를 실행한다.

    중요: QUBO의 **그래프 구조는 lambda와 무관**하다. lambda는 계수의 크기만
    바꿀 뿐 어떤 변수쌍이 연결되는지는 바꾸지 않는다. 따라서 embedding을
    한 번만 계산해 모든 lambda에 재사용한다. 이렇게 해야 관측된 차이가
    embedding 운에 의한 것이 아니라 lambda 때문임을 보장할 수 있다.
    """
    from dwave.system import FixedEmbeddingComposite

    from . import qa_solver

    points: list[LambdaPoint] = []
    try:
        sampler = qa_solver.get_sampler(qa_config)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"QPU sampler를 생성하지 못했습니다: {error}. "
            f"DWAVE_API_TOKEN을 설정했는지 확인하십시오."
        ) from error

    # 그래프 구조는 lambda와 무관하므로 embedding은 한 번만 계산한다.
    base_model = build_qubo(instance, formulation, 1.1, precision)
    embedding_result = qa_solver.try_embedding(base_model, sampler, embedding_config)
    if embedding_result.status != qa_solver.STATUS_OK:
        raise RuntimeError(
            f"embedding에 실패했습니다: {embedding_result.message}. "
            f"lambda 실험은 embedding이 가능한 조합에서만 의미가 있습니다."
        )
    composite = FixedEmbeddingComposite(sampler, embedding_result.embedding)

    for multiplier, lam in lambda_grid(reference_objective, multipliers):
        model = build_qubo(instance, formulation, 1.1, precision, lambda_value=lam)
        max_abs, scaled = scaled_objective_coefficient(model, instance)
        bqm = sa_solver.to_bqm(model)
        chain_strength = qa_solver.compute_chain_strength(
            bqm, float(qa_config["chain_strength_alpha"])
        )
        import time

        start = time.perf_counter()
        sampleset = composite.sample(
            bqm,
            num_reads=int(qa_config["num_reads"]),
            annealing_time=float(qa_config["annealing_time"]),
            chain_strength=chain_strength,
            answer_mode="raw",
        )
        sampleset.resolve()
        runtime = time.perf_counter() - start

        chain_break = (
            float(np.mean(sampleset.record.chain_break_fraction))
            if "chain_break_fraction" in sampleset.record.dtype.names
            else float("nan")
        )
        outcome = analyze_sampleset(
            model=model,
            instance=instance,
            sampleset=sampleset,
            solver="QA",
            runtime=runtime,
            num_reads=int(qa_config["num_reads"]),
            tolerance=tolerance,
        )
        stats = model.coefficient_stats()
        gap = (
            (outcome.best_feasible_objective - reference_objective)
            / abs(reference_objective) * 100.0
            if outcome.best_feasible_objective is not None
            else float("nan")
        )
        points.append(
            LambdaPoint(
                multiplier=multiplier,
                lambda_value=lam,
                qubo_min=stats["qubo_min"],
                qubo_max=stats["qubo_max"],
                qubo_range=stats["qubo_range"],
                max_abs_coupling=max_abs,
                scaled_objective_coefficient=scaled,
                solver="QA",
                status=outcome.status,
                feasible_fraction=outcome.feasible_fraction,
                best_feasible_objective=outcome.best_feasible_objective,
                true_gap_percent=gap,
                ground_state_is_feasible=check_ground_state_feasible(
                    model, instance, reference_sample, outcome
                ),
                runtime=runtime,
                extra={
                    "qa_chain_strength": chain_strength,
                    "qa_chain_break_fraction": chain_break,
                    "physical_qubits": embedding_result.num_physical_qubits,
                    "max_chain_length": embedding_result.max_chain_length,
                },
            )
        )
    return points


def interpret(frame: Any) -> list[str]:
    """SA와 QA의 lambda 반응을 비교해 가설을 구분한다.

    Args:
        frame: multiplier, solver, feasible_fraction 컬럼을 갖는 DataFrame.

    Returns:
        해석 문장 목록. 데이터가 부족하면 그 사실을 알린다.
    """
    lines: list[str] = []
    for solver in ("SA", "QA"):
        subset = frame[frame["solver"] == solver]
        if subset.empty:
            lines.append(f"{solver}: 결과 없음")
            continue
        low = subset[subset["multiplier"] <= 1.0]["feasible_fraction"].mean()
        high = subset[subset["multiplier"] >= 3.0]["feasible_fraction"].mean()
        lines.append(
            f"{solver}: lambda 낮은 구간 feasible {low:.3f} vs "
            f"높은 구간 {high:.3f} (차이 {low - high:+.3f})"
        )

    sa = frame[frame["solver"] == "SA"]
    qa = frame[frame["solver"] == "QA"]
    if sa.empty or qa.empty:
        lines.append(
            "가설 구분에는 SA와 QA 결과가 모두 필요합니다. "
            "QA는 DWAVE_API_TOKEN이 설정된 환경에서 실행하십시오."
        )
        return lines

    def response(subset: Any) -> float:
        low = subset[subset["multiplier"] <= 1.0]["feasible_fraction"].mean()
        high = subset[subset["multiplier"] >= 3.0]["feasible_fraction"].mean()
        return float(low - high)

    sa_delta, qa_delta = response(sa), response(qa)
    if qa_delta > 0.05 and sa_delta <= 0.05:
        lines.append("=> QA만 lambda에 반응. 가설 A(precision)를 지지합니다.")
    elif sa_delta > 0.05 and qa_delta > 0.05:
        lines.append("=> SA와 QA가 함께 반응. 가설 B(landscape) 기여가 큽니다.")
    elif sa_delta <= 0.05 and qa_delta <= 0.05:
        lines.append("=> 둘 다 반응 없음. lambda 외의 요인을 찾아야 합니다.")
    else:
        lines.append("=> SA만 반응. 예상 밖 패턴이므로 원자료를 직접 확인하십시오.")
    return lines
