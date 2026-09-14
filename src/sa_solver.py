"""Simulated Annealing(dwave-neal)으로 QUBO를 해결하는 모듈.

SA는 QUBO를 그대로 입력받는다. QA와의 비교가 공정하도록 **동일한 QUBO**를
사용하며, 파라미터는 설정 파일에 고정하고 결과를 보고 튜닝하지 않는다.
"""

from __future__ import annotations

import time
from typing import Any

import dimod
import neal

from .data_generator import CFLPInstance
from .evaluation import SolverOutcome, analyze_sampleset
from .qubo_builder import QUBOModel


def to_bqm(model: QUBOModel) -> dimod.BinaryQuadraticModel:
    """QUBOModel을 dimod BinaryQuadraticModel로 변환한다.

    계수가 0이어서 QUBO dict에 나타나지 않는 변수도 반드시 BQM에 포함시켜야
    sample과 QUBO 변수 순서가 어긋나지 않는다.
    """
    bqm = dimod.BinaryQuadraticModel.from_qubo(model.to_dimod_dict(), model.offset)
    for name in model.variables:
        if name not in bqm.variables:
            bqm.add_variable(name, 0.0)
    return bqm


def solve(
    model: QUBOModel,
    instance: CFLPInstance,
    sa_config: dict[str, Any],
    tolerance: float,
) -> SolverOutcome:
    """QUBO를 SA로 해결하고 결과를 평가한다.

    Args:
        model: 대상 QUBO.
        instance: 원래 CFLP instance (decode 후 검증에 사용).
        sa_config: 설정 파일의 ``sa`` 섹션.
        tolerance: feasibility 판정 허용오차.

    Returns:
        ``SolverOutcome``.
    """
    bqm = to_bqm(model)
    num_reads = int(sa_config["num_reads"])
    num_sweeps = int(sa_config["num_sweeps"])
    seed = int(sa_config["seed"])

    sampler = neal.SimulatedAnnealingSampler()

    start = time.perf_counter()
    sampleset = sampler.sample(
        bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed
    )
    runtime = time.perf_counter() - start

    return analyze_sampleset(
        model=model,
        instance=instance,
        sampleset=sampleset,
        solver="SA",
        runtime=runtime,
        num_reads=num_reads,
        tolerance=tolerance,
        extra={"sa_num_sweeps": num_sweeps, "sa_seed": seed},
    )
