"""Constraint별 penalty coefficient 균형 실험 모듈.

배경
----
본 실험(notebook 04, 05)은 모든 constraint에 동일한 lambda를 사용했다.
그런데 constraint마다 계수의 크기가 다르므로, 같은 lambda를 써도
**하드웨어가 보는 제약 강도는 크게 달라진다.**

    capacity  : sum_i d_i x_ij - s_j y_j + slack = 0
                계수에 d_i, s_j가 들어 있어 제곱 시 s_j^2 규모로 증폭된다.
    assignment: sum_j x_ij - 1 = 0
                계수가 모두 1이므로 lambda가 그대로 남는다.

그 결과 두 penalty의 크기 비는 대략 2 / s_max^2 이 되고, 8x8에서는
약 10^-4, 즉 assignment가 capacity보다 1만 배 약하다.

이 모듈은 capacity penalty를 고정한 채 assignment penalty에만 배수를
곱해가며(grid search) 그 불균형을 보정했을 때 무엇이 달라지는지 관찰한다.

원칙 위반이 아닌 이유
--------------------
penalty method가 요구하는 것은 "각 constraint의 위반이 objective 이득보다
비쌀 것", 즉 각 constraint에 대해 lambda_k > Z_ub 이다. **모든 constraint가
같은 lambda를 써야 한다는 요구는 없다.** assignment penalty를 키우는 것은
그 조건을 더 강하게 만들 뿐이므로 ground state가 feasible이라는 보장은
유지된다.

dry-run 모드
-----------
    qa_dryrun=True  : QPU 없이 오프라인으로 수행한다.
                      - embedding: dwave_networkx의 Pegasus P16 그래프에
                        minorminer를 직접 실행한다(실제 토폴로지).
                      - sampling: Simulated Annealing으로 대체한다.
                      - **노이즈를 주지 않는다.** 즉 이상적 조건에서의
                        상한을 본다.
    qa_dryrun=False : 실제 QPU에 접속한다.

두 모드의 차이가 곧 하드웨어가 잃는 양이다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import sa_solver
from .data_generator import CFLPInstance
from .evaluation import analyze_sampleset
from .qubo_builder import QUBOModel, build_qubo

# assignment/capacity 비율 grid의 기본값. 로그 간격으로 훑되 1.0(기존 설정)을
# 반드시 포함한다.
DEFAULT_RATIOS: tuple[float, ...] = (
    1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0,
)

# Pegasus P16 = Advantage 세대의 토폴로지. 결함 없는 이상적 그래프이다.
PEGASUS_SIZE: int = 16


@dataclass
class BalanceRecord:
    """ratio grid 한 점의 관측 결과."""

    instance: str
    formulation: str
    mode: str
    capacity_multiplier: float
    ratio: float
    lambda_capacity: float
    lambda_assignment: float
    qubo_variables: int
    qubo_quadratic_terms: int
    qubo_range: float
    max_abs_coupling: float
    assignment_strength: float
    capacity_strength: float
    objective_strength: float
    status: str
    feasible_fraction: float
    best_feasible_objective: float | None
    true_gap_percent: float
    runtime: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = {
            key: getattr(self, key)
            for key in (
                "instance", "formulation", "mode", "capacity_multiplier", "ratio",
                "lambda_capacity", "lambda_assignment", "qubo_variables",
                "qubo_quadratic_terms", "qubo_range", "max_abs_coupling",
                "assignment_strength", "capacity_strength", "objective_strength",
                "status", "feasible_fraction", "best_feasible_objective",
                "true_gap_percent", "runtime",
            )
        }
        record.update(self.extra)
        return record


def balanced_ratio(instance: CFLPInstance) -> float:
    """assignment penalty가 capacity penalty와 같아지는 배수.

    capacity의 최대 계수는 lambda * s_max^2, assignment의 최대 계수는
    2 * lambda * ratio 이므로 ratio = s_max^2 / 2 에서 둘이 같아진다.
    이 값을 넘으면 assignment가 최대 계수가 되어 **coefficient range가
    커지기 시작한다.** 즉 이 값이 "공짜로 올릴 수 있는" 상한이다.
    """
    return float(int(instance.capacities.max()) ** 2) / 2.0


def coefficient_strengths(
    model: QUBOModel, instance: CFLPInstance
) -> dict[str, float]:
    """auto_scale 이후 각 항이 어느 정도 크기로 보이는지 계산한다.

    D-Wave는 모든 계수를 max|J|로 나누어 하드웨어 범위에 맞춘다. ICE 노이즈는
    정규화 단위로 대략 0.01~0.03이므로, 이 값들과 직접 비교할 수 있다.
    """
    bqm = sa_solver.to_bqm(model)
    couplings = [abs(value) for value in bqm.spin.quadratic.values()]
    max_abs = max(couplings) if couplings else 1.0

    lam_capacity = model.penalty.value
    lam_assignment = float(model.metadata.get("assignment_weight", lam_capacity))
    demand_max = float(instance.demands.max())
    weighted = instance.transport_costs * instance.demands[:, None]

    return {
        "max_abs_coupling": float(max_abs),
        # assignment/demand penalty의 대표 교차항
        "assignment_strength": float(2.0 * lam_assignment / max_abs),
        # capacity penalty의 대표 교차항 (x-x 항)
        "capacity_strength": float(
            2.0 * lam_capacity * demand_max * demand_max / max_abs
        ),
        # objective 중 가장 작은 계수
        "objective_strength": float(min(weighted.min(), instance.fixed_costs.min()) / max_abs),
    }


def _find_offline_embedding(
    model: QUBOModel, embedding_config: dict[str, Any]
) -> dict[str, Any]:
    """Pegasus P16 그래프에 오프라인으로 embedding을 시도한다."""
    import dwave_networkx as dnx
    import minorminer

    bqm = sa_solver.to_bqm(model)
    source = [(u, v) for u, v in bqm.quadratic]
    target = list(dnx.pegasus_graph(PEGASUS_SIZE).edges())

    start = time.perf_counter()
    embedding = minorminer.find_embedding(
        source,
        target,
        random_seed=int(embedding_config["seed"]),
        tries=int(embedding_config["tries"]),
        timeout=int(embedding_config["timeout"]),
    )
    elapsed = time.perf_counter() - start
    if not embedding:
        return {
            "embedding_status": "NOT_EMBEDDABLE",
            "physical_qubits": 0,
            "max_chain_length": 0,
            "embedding_search_time": elapsed,
        }
    chains = [len(chain) for chain in embedding.values()]
    return {
        "embedding_status": "OK",
        "physical_qubits": int(sum(chains)),
        "max_chain_length": int(max(chains)),
        "mean_chain_length": float(np.mean(chains)),
        "embedding_search_time": elapsed,
    }


def run_ratio_sweep(
    instance: CFLPInstance,
    formulation: str,
    reference_objective: float,
    config: dict[str, Any],
    capacity_multiplier: float,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    qa_dryrun: bool = True,
    check_embedding: bool = True,
) -> list[BalanceRecord]:
    """assignment/capacity penalty 비율을 훑는다.

    Args:
        capacity_multiplier: capacity penalty에 쓸 배수. U_obj에 곱해지며,
            기존 실험의 ``config['penalty']['margin']``과 같은 의미이다.
        ratios: assignment penalty가 capacity penalty의 몇 배인지.
        qa_dryrun: True면 오프라인(Pegasus embedding + SA, 노이즈 없음),
            False면 실제 QPU.
        check_embedding: dry-run에서 embedding을 확인할지 여부.
            grid의 모든 점이 같은 그래프를 갖기 때문에 한 번만 계산한다.

    Returns:
        ratio별 ``BalanceRecord`` 목록.
    """
    tolerance = float(config["feasibility"]["tolerance"])
    precision = int(config["encoding"]["precision"])
    mode = "offline" if qa_dryrun else "qpu"

    sampler = None
    composite = None
    embedding_info: dict[str, Any] = {}

    # 그래프 구조는 penalty 값과 무관하므로 embedding은 한 번만 계산한다.
    base_model = build_qubo(instance, formulation, capacity_multiplier, precision)
    if qa_dryrun:
        if check_embedding:
            embedding_info = _find_offline_embedding(base_model, config["embedding"])
    else:
        from dwave.system import FixedEmbeddingComposite

        from . import qa_solver

        sampler = qa_solver.get_sampler(config["qa"])
        result = qa_solver.try_embedding(base_model, sampler, config["embedding"])
        embedding_info = result.to_record()
        if result.status != qa_solver.STATUS_OK:
            raise RuntimeError(
                f"embedding에 실패했습니다: {result.message}. "
                f"이 조합에서는 QPU 실험이 불가능합니다."
            )
        composite = FixedEmbeddingComposite(sampler, result.embedding)

    records: list[BalanceRecord] = []
    for ratio in ratios:
        model = build_qubo(
            instance,
            formulation,
            capacity_multiplier,
            precision,
            assignment_multiplier=float(ratio),
        )
        strengths = coefficient_strengths(model, instance)
        stats = model.coefficient_stats()

        if qa_dryrun:
            outcome = sa_solver.solve(model, instance, config["sa"], tolerance)
            extra: dict[str, Any] = {"sampler": "SimulatedAnnealing"}
        else:
            from . import qa_solver

            bqm = sa_solver.to_bqm(model)
            chain_strength = qa_solver.compute_chain_strength(
                bqm, float(config["qa"]["chain_strength_alpha"])
            )
            start = time.perf_counter()
            sampleset = composite.sample(
                bqm,
                num_reads=int(config["qa"]["num_reads"]),
                annealing_time=float(config["qa"]["annealing_time"]),
                chain_strength=chain_strength,
                answer_mode="raw",
            )
            sampleset.resolve()
            elapsed = time.perf_counter() - start
            outcome = analyze_sampleset(
                model=model,
                instance=instance,
                sampleset=sampleset,
                solver="QA",
                runtime=elapsed,
                num_reads=int(config["qa"]["num_reads"]),
                tolerance=tolerance,
            )
            chain_break = (
                float(np.mean(sampleset.record.chain_break_fraction))
                if "chain_break_fraction" in sampleset.record.dtype.names
                else float("nan")
            )
            extra = {
                "sampler": "DWaveSampler",
                "qa_chain_strength": chain_strength,
                "qa_chain_break_fraction": chain_break,
            }

        extra.update(embedding_info)
        gap = (
            (outcome.best_feasible_objective - reference_objective)
            / abs(reference_objective) * 100.0
            if outcome.best_feasible_objective is not None
            else float("nan")
        )
        records.append(
            BalanceRecord(
                instance=instance.name,
                formulation=formulation,
                mode=mode,
                capacity_multiplier=float(capacity_multiplier),
                ratio=float(ratio),
                lambda_capacity=model.penalty.value,
                lambda_assignment=float(model.metadata["assignment_weight"]),
                qubo_variables=model.num_variables,
                qubo_quadratic_terms=model.num_quadratic_terms,
                qubo_range=stats["qubo_range"],
                max_abs_coupling=strengths["max_abs_coupling"],
                assignment_strength=strengths["assignment_strength"],
                capacity_strength=strengths["capacity_strength"],
                objective_strength=strengths["objective_strength"],
                status=outcome.status,
                feasible_fraction=outcome.feasible_fraction,
                best_feasible_objective=outcome.best_feasible_objective,
                true_gap_percent=gap,
                runtime=outcome.runtime,
                extra=extra,
            )
        )
    return records


def summarize(frame: Any) -> list[str]:
    """ratio 1(기존 설정) 대비 최선의 ratio가 얼마나 나아졌는지 정리한다."""
    lines: list[str] = []
    for (instance, formulation, mode), group in frame.groupby(
        ["instance", "formulation", "mode"], sort=False
    ):
        baseline = group[group["ratio"] == 1.0]
        if baseline.empty:
            lines.append(f"{instance} {formulation} ({mode}): ratio=1 결과가 없습니다.")
            continue
        base_feasible = float(baseline["feasible_fraction"].iloc[0])
        best = group.loc[group["feasible_fraction"].idxmax()]
        improvement = (
            f"{float(best['feasible_fraction']) / base_feasible:.0f}배"
            if base_feasible > 0
            else "0에서 상승"
        )
        lines.append(
            f"{instance} {formulation} ({mode}): "
            f"feasible {base_feasible * 100:.1f}% -> "
            f"{float(best['feasible_fraction']) * 100:.1f}% "
            f"(ratio={best['ratio']:.0f}, {improvement}), "
            f"range {float(baseline['qubo_range'].iloc[0]):.2e} -> "
            f"{float(best['qubo_range']):.2e}"
        )
    return lines
