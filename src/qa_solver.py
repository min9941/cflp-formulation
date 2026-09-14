"""D-Wave QPU(Quantum Annealing)로 QUBO를 해결하는 모듈.

원칙
----
* QA parameter tuning은 이 연구의 목적이 아니다. 모든 파라미터는 설정
  파일에 고정하며 실험 도중 결과를 보고 변경하지 않는다.
* chain strength는 절대값이 아니라 **계수에 비례하는 규칙**으로 고정한다.

      chain_strength = alpha * max_ij |J_ij|

  여기서 J는 Ising 표현의 coupling이다. instance마다 QUBO 계수 스케일이
  크게 다르므로, 절대값을 고정하면 오히려 instance 간 비교 조건이
  불공정해진다. 규칙을 고정하는 것이 "튜닝하지 않는다"는 원칙에 부합한다.
* embedding은 seed와 timeout을 고정하여 재현 가능하게 한다.
* embedding에 실패하면 formulation이나 QUBO를 축소하지 않고
  ``NOT_EMBEDDABLE``로 기록한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_generator import CFLPInstance
from .evaluation import SolverOutcome, analyze_sampleset, failed_outcome
from .qubo_builder import QUBOModel
from .sa_solver import to_bqm

STATUS_NOT_EMBEDDABLE: str = "NOT_EMBEDDABLE"
STATUS_NO_QPU: str = "NO_QPU_ACCESS"
STATUS_OK: str = "OK"


@dataclass
class EmbeddingResult:
    """minor embedding 시도 결과."""

    status: str
    embedding: dict[Any, list[int]] | None
    num_logical_variables: int
    num_physical_qubits: int
    max_chain_length: int
    mean_chain_length: float
    search_time: float
    message: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "embedding_status": self.status,
            "logical_variables": self.num_logical_variables,
            "physical_qubits": self.num_physical_qubits,
            "max_chain_length": self.max_chain_length,
            "mean_chain_length": self.mean_chain_length,
            "embedding_search_time": self.search_time,
        }


def compute_chain_strength(bqm: Any, alpha: float) -> float:
    """chain strength = alpha * max|J| 규칙으로 계산한다.

    QUBO를 Ising 표현으로 바꾼 뒤 coupling J의 최대 절댓값을 사용한다.
    """
    ising = bqm.spin
    couplings = [abs(value) for value in ising.quadratic.values()]
    max_coupling = max(couplings) if couplings else 1.0
    return float(alpha) * float(max_coupling)


def get_sampler(qa_config: dict[str, Any]) -> Any:
    """DWaveSampler를 생성한다.

    토큰이 없거나 접속에 실패하면 예외 대신 None을 반환하도록 하지 않고,
    호출부에서 상태를 기록할 수 있게 예외를 그대로 올린다.
    """
    from dwave.system import DWaveSampler

    solver_name = qa_config.get("solver_name")
    if solver_name:
        return DWaveSampler(solver=solver_name)
    return DWaveSampler()


def try_embedding(
    model: QUBOModel,
    sampler: Any,
    embedding_config: dict[str, Any],
) -> EmbeddingResult:
    """QUBO를 QPU 그래프에 minor embedding할 수 있는지 확인한다.

    seed/timeout/tries를 고정하여 판정이 재현 가능하도록 한다.
    """
    import minorminer

    bqm = to_bqm(model)
    source_edges = [(u, v) for u, v in bqm.quadratic]
    # 고립 변수도 embedding 대상에 포함되도록 self-node로 등록한다.
    source_nodes = list(bqm.variables)

    target_edges = sampler.edgelist

    start = time.perf_counter()
    embedding = minorminer.find_embedding(
        source_edges if source_edges else [(node, node) for node in source_nodes],
        target_edges,
        random_seed=int(embedding_config["seed"]),
        timeout=int(embedding_config["timeout"]),
        tries=int(embedding_config["tries"]),
    )
    search_time = time.perf_counter() - start

    if not embedding:
        return EmbeddingResult(
            status=STATUS_NOT_EMBEDDABLE,
            embedding=None,
            num_logical_variables=model.num_variables,
            num_physical_qubits=0,
            max_chain_length=0,
            mean_chain_length=float("nan"),
            search_time=search_time,
            message=(
                f"QUBO 변수 {model.num_variables}개를 QPU 그래프에 embedding하지 "
                f"못했습니다 (seed={embedding_config['seed']}, "
                f"timeout={embedding_config['timeout']}s)."
            ),
        )

    chain_lengths = [len(chain) for chain in embedding.values()]
    return EmbeddingResult(
        status=STATUS_OK,
        embedding=embedding,
        num_logical_variables=model.num_variables,
        num_physical_qubits=int(sum(chain_lengths)),
        max_chain_length=int(max(chain_lengths)),
        mean_chain_length=float(np.mean(chain_lengths)),
        search_time=search_time,
        message="embedding 성공",
    )


def solve(
    model: QUBOModel,
    instance: CFLPInstance,
    qa_config: dict[str, Any],
    embedding_config: dict[str, Any],
    tolerance: float,
) -> tuple[SolverOutcome, EmbeddingResult | None]:
    """QUBO를 QA로 해결한다.

    Returns:
        (SolverOutcome, EmbeddingResult). QPU에 접근할 수 없거나 embedding에
        실패하면 outcome.status에 사유가 기록되고 결과 값들은 NaN이 된다.
    """
    from dwave.system import FixedEmbeddingComposite

    try:
        sampler = get_sampler(qa_config)
    except Exception as error:  # noqa: BLE001 - 접속 실패 사유를 그대로 기록
        return (
            failed_outcome(
                "QA",
                STATUS_NO_QPU,
                extra={"qa_message": f"QPU sampler 생성 실패: {error}"},
            ),
            None,
        )

    embedding_result = try_embedding(model, sampler, embedding_config)
    if embedding_result.status != STATUS_OK:
        return (
            failed_outcome(
                "QA",
                STATUS_NOT_EMBEDDABLE,
                extra={
                    "qa_message": embedding_result.message,
                    **embedding_result.to_record(),
                },
            ),
            embedding_result,
        )

    bqm = to_bqm(model)
    chain_strength = compute_chain_strength(bqm, float(qa_config["chain_strength_alpha"]))
    num_reads = int(qa_config["num_reads"])
    annealing_time = float(qa_config["annealing_time"])
    chain_break_method = str(qa_config.get("chain_break_method", "majority_vote"))

    composite = FixedEmbeddingComposite(sampler, embedding_result.embedding)

    start = time.perf_counter()
    try:
        sampleset = composite.sample(
            bqm,
            num_reads=num_reads,
            annealing_time=annealing_time,
            chain_strength=chain_strength,
            answer_mode="raw",
        )
        sampleset.resolve()
    except Exception as error:  # noqa: BLE001
        return (
            failed_outcome(
                "QA",
                "QPU_RUN_FAILED",
                extra={
                    "qa_message": f"QPU 실행 실패: {error}",
                    **embedding_result.to_record(),
                },
            ),
            embedding_result,
        )
    wall_clock = time.perf_counter() - start

    timing = dict(sampleset.info.get("timing", {}))
    chain_break_fraction = float(
        np.mean(sampleset.record.chain_break_fraction)
    ) if "chain_break_fraction" in sampleset.record.dtype.names else float("nan")

    extra: dict[str, Any] = {
        "qa_chain_strength": chain_strength,
        "qa_chain_strength_alpha": float(qa_config["chain_strength_alpha"]),
        "qa_annealing_time": annealing_time,
        "qa_chain_break_method": chain_break_method,
        "qa_chain_break_fraction": chain_break_fraction,
        "qa_wall_clock": wall_clock,
        "qa_qpu_access_time_us": float(timing.get("qpu_access_time", float("nan"))),
        "qa_qpu_sampling_time_us": float(timing.get("qpu_sampling_time", float("nan"))),
    }
    extra.update(embedding_result.to_record())

    outcome = analyze_sampleset(
        model=model,
        instance=instance,
        sampleset=sampleset,
        solver="QA",
        runtime=wall_clock,
        num_reads=num_reads,
        tolerance=tolerance,
        extra=extra,
    )
    return outcome, embedding_result
