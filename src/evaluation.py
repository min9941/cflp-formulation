"""QUBO sample 평가 및 성능 지표 계산 모듈.

핵심 원칙
---------
* QUBO energy만 비교하지 않는다. 모든 해는 원래 CFLP 변수 공간으로
  decode한 뒤 원래 objective를 다시 계산한다.
* feasibility는 원래 CFLP formulation 기준으로 재검증한다.
* optimality gap은 Gurobi가 구한 true optimum을 기준으로 계산한다.

sample-energy 짝맞춤에 대한 주의
--------------------------------
dimod ``SampleSet``에서 ``.samples()``는 정렬된 뷰를 반환하므로
``record.energy``와 인덱스가 어긋날 수 있다. 본 모듈은 항상
``sampleset.record.sample``과 ``sampleset.record.energy``를 같은
인덱스로 함께 읽어 이 문제를 원천적으로 차단한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import cflp_ms, cflp_ss
from .data_generator import CFLPInstance
from .decoder import decode
from .qubo_builder import QUBOModel


@dataclass
class SampleEvaluation:
    """하나의 sample에 대한 평가 결과."""

    energy: float
    objective: float
    is_feasible: bool
    max_violation: float
    total_violation: float
    sample: np.ndarray
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class SolverOutcome:
    """SA 또는 QA 실행 결과 전체."""

    solver: str
    status: str
    runtime: float
    num_reads: int
    best_energy: float
    best_objective: float
    best_is_feasible: bool
    best_total_violation: float
    best_max_violation: float
    feasible_fraction: float
    best_feasible_objective: float | None
    best_feasible_energy: float | None
    num_samples: int
    best_sample: np.ndarray | None = None
    best_feasible_sample: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """결과 표 저장용 dict (sample 배열은 제외)."""
        record: dict[str, Any] = {
            "solver": self.solver,
            "status": self.status,
            "runtime": self.runtime,
            "num_reads": self.num_reads,
            "best_energy": self.best_energy,
            "best_objective": self.best_objective,
            "best_is_feasible": self.best_is_feasible,
            "best_total_violation": self.best_total_violation,
            "best_max_violation": self.best_max_violation,
            "feasible_fraction": self.feasible_fraction,
            "best_feasible_objective": self.best_feasible_objective,
            "best_feasible_energy": self.best_feasible_energy,
            "num_samples": self.num_samples,
        }
        record.update(self.extra)
        return record


def evaluate_sample(
    model: QUBOModel,
    instance: CFLPInstance,
    sample: np.ndarray,
    tolerance: float,
    energy: float | None = None,
) -> SampleEvaluation:
    """하나의 binary sample을 decode하여 원래 문제 기준으로 평가한다.

    Args:
        energy: 미리 계산해 둔 QUBO energy. None이면 이 함수에서 계산한다.
            여러 sample을 처리할 때는 ``model.energies``로 일괄 계산한 값을
            넘겨 반복문 비용을 줄인다.
    """
    formulation = model.metadata["formulation"]
    solution = decode(model, sample)
    if formulation == "SS":
        objective = cflp_ss.evaluate_objective(instance, solution)
        report = cflp_ss.check_feasibility(instance, solution, tolerance)
    else:
        objective = cflp_ms.evaluate_objective(instance, solution)
        report = cflp_ms.check_feasibility(
            instance, solution, tolerance, require_integral=True
        )
    return SampleEvaluation(
        energy=float(model.energy(sample)) if energy is None else float(energy),
        objective=objective,
        is_feasible=report.is_feasible,
        max_violation=report.max_violation,
        total_violation=report.total_violation,
        sample=np.asarray(sample, dtype=int),
        details=report.details,
    )


def _sampleset_to_arrays(
    model: QUBOModel, sampleset: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SampleSet에서 (samples, energies, occurrences)를 안전하게 추출한다.

    ``record``를 직접 사용하여 sample과 energy의 인덱스가 어긋나지 않도록 한다.
    변수 순서는 ``sampleset.variables``를 QUBO 변수 순서로 재배열한다.
    """
    record = sampleset.record
    variable_order = list(sampleset.variables)
    position = {name: index for index, name in enumerate(variable_order)}

    missing = [name for name in model.variables if name not in position]
    if missing:
        raise ValueError(
            f"SampleSet에 QUBO 변수가 누락되었습니다 (예: {missing[:3]}). "
            f"고립 변수(isolated variable) 처리 여부를 확인하십시오."
        )

    columns = [position[name] for name in model.variables]
    samples = np.asarray(record.sample, dtype=int)[:, columns]
    energies = np.asarray(record.energy, dtype=float)
    occurrences = np.asarray(
        getattr(record, "num_occurrences", np.ones(len(energies))), dtype=int
    )
    return samples, energies, occurrences


def analyze_sampleset(
    model: QUBOModel,
    instance: CFLPInstance,
    sampleset: Any,
    solver: str,
    runtime: float,
    num_reads: int,
    tolerance: float,
    extra: dict[str, Any] | None = None,
) -> SolverOutcome:
    """SampleSet 전체를 평가하여 ``SolverOutcome``으로 정리한다.

    best sample 하나뿐 아니라 feasible sample 비율 등 안정성 지표도 함께
    기록한다.
    """
    samples, energies, occurrences = _sampleset_to_arrays(model, sampleset)

    # energy는 일괄 계산하여 sample과 같은 인덱스로 사용한다.
    computed_energies = model.energies(samples)
    evaluations = [
        evaluate_sample(
            model, instance, samples[row], tolerance, energy=computed_energies[row]
        )
        for row in range(samples.shape[0])
    ]

    total_reads = int(occurrences.sum())
    feasible_reads = int(
        sum(
            occurrence
            for occurrence, evaluation in zip(occurrences, evaluations)
            if evaluation.is_feasible
        )
    )

    best_index = int(np.argmin(energies))
    best = evaluations[best_index]

    feasible_pairs = [
        (evaluation.objective, evaluation)
        for evaluation in evaluations
        if evaluation.is_feasible
    ]
    if feasible_pairs:
        best_feasible = min(feasible_pairs, key=lambda pair: pair[0])[1]
        best_feasible_objective: float | None = best_feasible.objective
        best_feasible_energy: float | None = best_feasible.energy
        best_feasible_sample: np.ndarray | None = best_feasible.sample
    else:
        best_feasible_objective = None
        best_feasible_energy = None
        best_feasible_sample = None

    # ---- sample-energy 짝맞춤 교차검증 ----
    # sample과 energy의 인덱스가 어긋나면 모든 하위 지표가 조용히 오염되므로,
    # best 하나가 아니라 **모든 sample**에 대해 우리가 직접 계산한 energy와
    # sampler가 보고한 energy를 비교한다.
    #
    # 스케일 주의: penalty 항은 10^9 규모인데 energy는 10^4 규모까지 상쇄된다.
    # 따라서 energy 자체로 나누면 순수 부동소수점 오차도 크게 보인다. QUBO
    # 계수의 최대 절댓값을 스케일에 포함하여 "실제 합산 정밀도" 기준으로 본다.
    reported_energy = float(energies[best_index])
    coefficient_scale = max(
        abs(value) for value in model.coefficient_stats().values()
    )
    energy_scale = max(
        1.0, abs(best.energy), abs(reported_energy), coefficient_scale
    )
    energy_mismatch = float(
        np.abs(computed_energies - energies).max(initial=0.0) / energy_scale
    )
    # 순위가 뒤바뀌었는지도 확인한다. 짝맞춤 오류의 가장 직접적인 증거다.
    argmin_agreement = bool(
        int(np.argmin(computed_energies)) == int(np.argmin(energies))
    )

    outcome_extra: dict[str, Any] = {
        "energy_mismatch": energy_mismatch,
        "reported_best_energy": reported_energy,
        "argmin_agreement": argmin_agreement,
        "unique_samples": int(samples.shape[0]),
        "feasible_reads": feasible_reads,
    }
    if extra:
        outcome_extra.update(extra)

    return SolverOutcome(
        solver=solver,
        status="OK",
        runtime=runtime,
        num_reads=num_reads,
        best_energy=best.energy,
        best_objective=best.objective,
        best_is_feasible=best.is_feasible,
        best_total_violation=best.total_violation,
        best_max_violation=best.max_violation,
        feasible_fraction=(feasible_reads / total_reads) if total_reads else 0.0,
        best_feasible_objective=best_feasible_objective,
        best_feasible_energy=best_feasible_energy,
        num_samples=int(samples.shape[0]),
        best_sample=best.sample,
        best_feasible_sample=best_feasible_sample,
        extra=outcome_extra,
    )


def failed_outcome(
    solver: str, status: str, extra: dict[str, Any] | None = None
) -> SolverOutcome:
    """실행하지 못한 경우의 결과 객체 (예: NOT_EMBEDDABLE)."""
    return SolverOutcome(
        solver=solver,
        status=status,
        runtime=float("nan"),
        num_reads=0,
        best_energy=float("nan"),
        best_objective=float("nan"),
        best_is_feasible=False,
        best_total_violation=float("nan"),
        best_max_violation=float("nan"),
        feasible_fraction=float("nan"),
        best_feasible_objective=None,
        best_feasible_energy=None,
        num_samples=0,
        extra=extra or {},
    )


def true_optimality_gap(
    solver_objective: float | None, reference_objective: float
) -> float:
    """Gurobi true optimum 대비 gap(%)을 계산한다.

    Gap = (Obj_solver - Obj_Gurobi) / |Obj_Gurobi| * 100
    """
    if solver_objective is None or not np.isfinite(solver_objective):
        return float("nan")
    if reference_objective == 0.0:
        raise ZeroDivisionError("reference objective가 0이어서 gap을 계산할 수 없습니다.")
    return (solver_objective - reference_objective) / abs(reference_objective) * 100.0


def energy_gap(solver_energy: float, reference_energy: float) -> float:
    """reference 해의 QUBO energy 대비 energy 차이."""
    if not np.isfinite(solver_energy) or not np.isfinite(reference_energy):
        return float("nan")
    return float(solver_energy - reference_energy)
