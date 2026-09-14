"""QUBO 변환의 정확성을 자동으로 검증하는 모듈.

SA/QA를 실행하기 전에 반드시 통과해야 하는 검증들이다.

검증 1: energy identity (random validation)
-------------------------------------------
무작위 binary 할당 z에 대해 다음 항등식이 성립해야 한다.

    QUBO_energy(z) == original_objective(decode(z)) + lambda * sum_k g_k(z)^2

이 식이 모든 무작위 샘플에서 성립하면, 목적함수 항과 penalty 항이
전개/합산 과정에서 손상되지 않았음을 의미한다. 상수 offset이 누락되면
이 검증에서 즉시 드러난다.

검증 2: reference solution 검증
-------------------------------
Gurobi 최적해를 QUBO 변수로 encoding했을 때
    - 모든 제약 잔차 g_k = 0
    - QUBO energy == Gurobi 목적값
이어야 한다.

검증 3: exhaustive validation (toy instance 전용)
-------------------------------------------------
실제 실험 instance는 변수 수가 커서 전수 열거가 불가능하다.
따라서 수요/용량을 매우 작게 고정한 별도의 2x2 toy instance에서만
전수 열거를 수행하여 QUBO 최소값이 MILP 최적값과 일치하는지 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import cflp_ms, cflp_ss
from .data_generator import CFLPInstance
from .decoder import constraint_residuals, decode, encode_solution
from .qubo_builder import QUBOModel, enumerate_all_samples

# 전수 열거를 허용하는 최대 변수 수 (2^24 = 약 1,600만)
MAX_EXHAUSTIVE_VARIABLES: int = 24


@dataclass
class ValidationResult:
    """검증 결과 하나."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.message}"


def _objective_function(
    model: QUBOModel,
) -> Callable[[CFLPInstance, Any], float]:
    """formulation에 맞는 원래 목적함수 계산 함수를 반환한다."""
    if model.metadata["formulation"] == "SS":
        return cflp_ss.evaluate_objective
    return cflp_ms.evaluate_objective


def reconstruct_energy(
    model: QUBOModel, instance: CFLPInstance, sample: np.ndarray
) -> tuple[float, float, float]:
    """decode 결과로부터 QUBO energy를 독립적으로 재구성한다.

    Returns:
        (원래 목적값, penalty 항, 재구성된 energy).
    """
    solution = decode(model, sample)
    objective = _objective_function(model)(instance, solution)
    residuals = constraint_residuals(model, instance, sample)
    penalty_term = float(model.penalty.value * float((residuals**2).sum()))
    return objective, penalty_term, objective + penalty_term


def validate_energy_identity(
    model: QUBOModel,
    instance: CFLPInstance,
    num_samples: int,
    seed: int,
    tolerance: float,
) -> ValidationResult:
    """무작위 샘플에 대해 energy 항등식을 검증한다."""
    rng = np.random.default_rng(seed)
    worst_error = 0.0
    worst_sample_index = -1

    samples = rng.integers(0, 2, size=(num_samples, model.num_variables))
    energies = model.energies(samples)

    for sample_index in range(num_samples):
        sample = samples[sample_index]
        energy = float(energies[sample_index])
        _, _, reconstructed = reconstruct_energy(model, instance, sample)
        # 계수 크기가 크므로 상대오차로 비교한다.
        scale = max(1.0, abs(energy), abs(reconstructed))
        error = abs(energy - reconstructed) / scale
        if error > worst_error:
            worst_error = error
            worst_sample_index = sample_index

    passed = worst_error <= tolerance
    message = (
        f"무작위 {num_samples}개 샘플의 최대 상대오차 {worst_error:.3e} "
        f"(허용 {tolerance:.1e})"
    )
    if not passed:
        message += f" — 샘플 #{worst_sample_index}에서 불일치"
    return ValidationResult(
        name=f"energy_identity[{model.metadata['formulation']}/{instance.name}]",
        passed=passed,
        message=message,
        details={"max_relative_error": worst_error, "num_samples": num_samples},
    )


def validate_reference_solution(
    model: QUBOModel,
    instance: CFLPInstance,
    solution: Any,
    reference_objective: float,
    tolerance: float,
) -> ValidationResult:
    """알려진 최적해를 QUBO로 encoding하여 energy가 목적값과 같은지 확인한다."""
    sample = encode_solution(model, instance, solution)
    residuals = constraint_residuals(model, instance, sample)
    max_residual = float(np.abs(residuals).max(initial=0.0))
    energy = model.energy(sample)
    scale = max(1.0, abs(reference_objective))
    error = abs(energy - reference_objective) / scale

    passed = max_residual <= tolerance and error <= tolerance
    message = (
        f"제약 잔차 최대 {max_residual:.3e}, "
        f"energy와 MILP 목적값의 상대차 {error:.3e}"
    )
    return ValidationResult(
        name=f"reference_encoding[{model.metadata['formulation']}/{instance.name}]",
        passed=passed,
        message=message,
        details={
            "max_residual": max_residual,
            "energy": energy,
            "reference_objective": reference_objective,
            "relative_error": error,
        },
    )


def validate_exhaustive(
    model: QUBOModel,
    instance: CFLPInstance,
    reference_objective: float,
    tolerance: float,
) -> ValidationResult:
    """모든 binary 할당을 전수 열거하여 QUBO 최소값을 확인한다.

    변수 수가 많아 열거가 불가능하면 검증을 건너뛰고 사유를 기록한다.
    """
    formulation = model.metadata["formulation"]
    if model.num_variables > MAX_EXHAUSTIVE_VARIABLES:
        return ValidationResult(
            name=f"exhaustive[{formulation}/{instance.name}]",
            passed=True,
            message=(
                f"SKIPPED — QUBO 변수 {model.num_variables}개는 전수 열거 한계"
                f"({MAX_EXHAUSTIVE_VARIABLES})를 초과합니다."
            ),
            details={"skipped": True, "num_variables": model.num_variables},
        )

    # 메모리 사용을 억제하기 위해 블록 단위로 나누어 전수 열거한다.
    num_variables = model.num_variables
    total = 1 << num_variables
    block_size = min(total, 1 << 16)
    bit_positions = np.arange(num_variables, dtype=np.int64)

    best_energy = float("inf")
    best_sample: np.ndarray | None = None
    for start in range(0, total, block_size):
        codes = np.arange(start, min(start + block_size, total), dtype=np.int64)
        block = ((codes[:, None] >> bit_positions[None, :]) & 1).astype(float)
        energies = model.energies(block)
        local_index = int(np.argmin(energies))
        if energies[local_index] < best_energy:
            best_energy = float(energies[local_index])
            best_sample = block[local_index].astype(int)

    if best_sample is None:
        raise RuntimeError("전수 열거 중 샘플을 하나도 평가하지 못했습니다.")

    solution = decode(model, best_sample)
    decoded_objective = _objective_function(model)(instance, solution)
    residuals = constraint_residuals(model, instance, best_sample)
    max_residual = float(np.abs(residuals).max(initial=0.0))

    scale = max(1.0, abs(reference_objective))
    error = abs(decoded_objective - reference_objective) / scale
    passed = max_residual <= tolerance and error <= tolerance

    message = (
        f"QUBO ground state의 목적값 {decoded_objective:.6f} vs "
        f"MILP 최적값 {reference_objective:.6f} (상대차 {error:.3e}), "
        f"제약 잔차 최대 {max_residual:.3e}"
    )
    return ValidationResult(
        name=f"exhaustive[{formulation}/{instance.name}]",
        passed=passed,
        message=message,
        details={
            "skipped": False,
            "num_variables": model.num_variables,
            "ground_state_energy": best_energy,
            "decoded_objective": decoded_objective,
            "reference_objective": reference_objective,
            "max_residual": max_residual,
        },
    )


def validate_encoding_completeness(upper_bounds: list[int]) -> ValidationResult:
    """binary expansion이 [0, U]의 모든 정수를 정확히 표현하는지 확인한다."""
    from .binary_encoder import build_encoding

    failures: list[str] = []
    for upper_bound in upper_bounds:
        encoding = build_encoding(upper_bound)
        representable = set()
        for code in range(1 << encoding.num_bits):
            bits = [(code >> position) & 1 for position in range(encoding.num_bits)]
            representable.add(encoding.decode(bits))
        expected = set(range(upper_bound + 1))
        if representable != expected:
            missing = sorted(expected - representable)
            extra = sorted(representable - expected)
            failures.append(f"U={upper_bound}: 누락 {missing[:5]}, 초과 {extra[:5]}")

    passed = not failures
    message = (
        f"{len(upper_bounds)}개 상한에 대해 표현 범위가 정확히 [0, U]와 일치"
        if passed
        else "; ".join(failures)
    )
    return ValidationResult(
        name="encoding_completeness",
        passed=passed,
        message=message,
        details={"num_checked": len(upper_bounds)},
    )


def summarize(results: list[ValidationResult]) -> dict[str, Any]:
    """검증 결과 요약."""
    passed = sum(1 for result in results if result.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "all_passed": passed == len(results),
    }
