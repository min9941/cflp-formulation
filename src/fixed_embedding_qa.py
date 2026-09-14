"""확보한 embedding으로 실제 QA를 실행하는 모듈 (notebook 30).

배경
----
본 실험(notebook 05)에서 MS 6x6은 timeout이 부족해 embedding에 실패했고
``NOT_EMBEDDABLE``로 기록되었다. 이후 embedding 실험(notebook 14)에서
Pegasus, tries=10, timeout=2400 조건으로 **5개 seed 모두 성공**했다.

이 모듈은 그 embedding을 고정해 실제 QA를 돌리고, chain break와 feasibility를
측정한다. 목적은 두 가지다.

1. notebook 05에서 비어 있던 MS 6x6 칸을 채운다.
2. 4x4 MS와 비교해 **chain break의 기여와 coefficient range의 기여를 분리**한다.
   4x4 MS는 embedding도 되고 chain break도 0.025%로 낮았는데 feasible 해가
   0개였다. 6x6은 chain이 훨씬 길므로(최대 31~40), 두 경우를 나란히 놓으면
   어느 요인이 지배적인지 가늠할 수 있다.

embedding 재사용
---------------
embedding mapping은 통계와 별도로 JSON에 저장해 재사용한다. 같은 solver,
같은 graph_id, 같은 seed/tries면 동일한 embedding이 재현되지만, working graph가
바뀌면 재현되지 않으므로 저장본을 쓰는 편이 안전하다. 저장본을 쓰기 전에는
``validate_embedding``으로 현재 그래프에서 유효한지 반드시 확인한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import embedding_study as ES
from . import qa_solver, sa_solver
from .data_generator import CFLPInstance
from .evaluation import SolverOutcome, analyze_sampleset, evaluate_sample
from .qubo_builder import QUBOModel, build_qubo

# seed를 고르는 기준. 값이 작을수록 좋은 지표들이다.
SELECTION_METRICS: tuple[str, ...] = (
    "physical_qubits",
    "avg_chain_length",
    "max_chain_length",
)


def select_seeds(
    results: pd.DataFrame,
    instance: str,
    formulation: str,
    topology: str,
    metrics: Sequence[str] = SELECTION_METRICS,
    max_seeds: int | None = None,
) -> pd.DataFrame:
    """embedding 품질 지표별 최선의 seed를 고른다.

    세 지표가 서로 다른 seed를 가리킬 수 있으므로 각 지표의 최선을 모두 뽑은 뒤
    중복을 제거한다. 물리 큐빗이 가장 적은 embedding과 최대 chain이 가장 짧은
    embedding이 다를 때, 어느 쪽이 QA에 유리한지는 미리 알 수 없기 때문이다.

    Args:
        results: embedding 실험 결과표 (notebook 14/15의 CSV).
        max_seeds: 상한. None이면 선정된 전부를 쓴다.

    Returns:
        선정된 seed의 행과 ``selected_by`` 컬럼(어느 지표로 뽑혔는지).
    """
    subset = results[
        (results["instance"] == instance)
        & (results["formulation"] == formulation)
        & (results["topology"] == topology)
        & (results["success"])
    ].copy()
    if subset.empty:
        raise ValueError(
            f"{instance} {formulation} {topology}에서 성공한 embedding이 "
            f"결과표에 없습니다. notebook 14를 먼저 실행하십시오."
        )

    chosen: dict[int, list[str]] = {}
    for metric in metrics:
        if metric not in subset:
            continue
        best = subset.loc[subset[metric].idxmin()]
        chosen.setdefault(int(best["seed"]), []).append(metric)

    rows = []
    for seed, reasons in chosen.items():
        row = subset[subset["seed"] == seed].iloc[0].to_dict()
        row["selected_by"] = ", ".join(reasons)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("physical_qubits")
    if max_seeds is not None:
        frame = frame.head(max_seeds)
    return frame


def obtain_embedding(
    instance: CFLPInstance,
    formulation: str,
    config: dict[str, Any],
    seed: int,
    tries: int,
    timeout: int,
    topology: str,
    embedding_dir: str | Path,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """embedding을 확보한다. 저장본이 있고 유효하면 재사용하고, 없으면 탐색한다.

    저장본을 쓰기 전에 현재 working graph에서 유효한지 검사한다. QPU의
    working graph는 교정 과정에서 바뀔 수 있고, 그러면 chain에 쓰인 물리 큐빗이
    사라지거나 연결이 끊겨 mapping을 쓸 수 없게 된다.
    """
    embedding_dir = Path(embedding_dir)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    name = f"{instance.name}_{formulation}_qpu-{topology}_seed{seed}_tries{tries}.json"
    path = embedding_dir / name

    def step(message: str, started: float | None = None) -> float:
        """진행 상황과 소요 시간을 찍는다. 어디서 오래 걸리는지 바로 보인다."""
        now = time.perf_counter()
        if started is not None:
            print(f"    [{now - started:6.1f}s] {message}", flush=True)
        else:
            print(f"    [      ] {message}", flush=True)
        return now

    clock = step("QUBO 그래프 생성 중")
    source, num_variables, num_edges = ES._source_graph(instance, formulation, config)
    clock = step(f"QUBO 그래프 완료 (변수 {num_variables}, 엣지 {num_edges})", clock)

    step("QPU 접속 중 (working graph 조회)")
    edges, actual_topology, solver_id = ES.qpu_target_edges(
        config["qa"], topology=topology
    )
    clock = step(f"QPU 접속 완료: {solver_id} (엣지 {len(edges)})", clock)

    target_hash = ES.graph_fingerprint(edges)
    clock = step("그래프 지문 계산 완료", clock)

    if path.exists():
        step(f"저장본 발견: {path.name}, 유효성 검사 중")
        embedding, meta = ES.load_embedding(path)
        check = ES.validate_embedding(
            {k: v for k, v in embedding.items()},
            [(str(u), str(v)) for u, v in source],
            edges,
        )
        clock = step("유효성 검사 완료", clock)
        if check["valid"]:
            meta["reused"] = True
            meta["solver_id_now"] = solver_id
            meta["target_hash_now"] = target_hash
            print(f"저장된 embedding 재사용: {path.name}")
            return embedding, meta
        print(
            f"저장본이 현재 그래프에서 유효하지 않습니다 "
            f"(사라진 큐빗 {check['num_missing_qubits']}, "
            f"끊긴 chain {check['num_broken_chains']}, "
            f"사라진 연결 {check['num_missing_links']}). 재탐색합니다."
        )

    step(
        f"embedding 재탐색 시작 (seed={seed}, tries={tries}, "
        f"timeout={timeout}s) — 수백~{timeout}초 걸릴 수 있습니다"
    )
    start = time.perf_counter()
    result = ES.run_trial(source, edges, timeout, tries, seed, return_embedding=True)
    if not result["success"]:
        raise RuntimeError(
            f"embedding 탐색에 실패했습니다 (seed={seed}, {time.perf_counter()-start:.0f}s). "
            f"notebook 14에서 성공했던 조건과 solver/graph가 같은지 확인하십시오."
        )

    embedding = result.pop("embedding")
    meta = {
        "instance": instance.name,
        "formulation": formulation,
        "topology": actual_topology,
        "topology_label": f"qpu:{actual_topology}",
        "solver_id": solver_id,
        "seed": int(seed),
        "tries": int(tries),
        "timeout": int(timeout),
        "source_hash": ES.graph_fingerprint(source),
        "target_hash": target_hash,
        "logical_variables": num_variables,
        "logical_edges": num_edges,
        "physical_qubits": result["physical_qubits"],
        "avg_chain_length": result["avg_chain_length"],
        "max_chain_length": result["max_chain_length"],
        "embedding_search_time": result["elapsed"],
        "reused": False,
    }
    ES.save_embedding(embedding, embedding_dir, meta)
    print(
        f"저장: {path.name}  "
        f"큐빗 {result['physical_qubits']}, 최대 chain {result['max_chain_length']}"
    )
    return {str(k): list(map(int, v)) for k, v in embedding.items()}, meta


def run_qa_with_embedding(
    model: QUBOModel,
    instance: CFLPInstance,
    embedding: dict[str, list[int]],
    config: dict[str, Any],
    chain_strength_alpha: float,
    solver_id: str | None = None,
    num_reads: int | None = None,
    annealing_time: float | None = None,
) -> SolverOutcome:
    """고정된 embedding으로 QA를 실행한다.

    Args:
        chain_strength_alpha: ``chain_strength = alpha * max|J|`` 규칙의 alpha.
            본 실험의 고정값은 1.5이다. 이 값을 바꾸는 것은 parameter tuning에
            해당하므로, 바꿀 경우 진단 실험으로 분류해 기록해야 한다.
        solver_id: 접속할 solver. embedding을 찾은 것과 **같은 solver**여야 한다.
    """
    from dwave.system import DWaveSampler, FixedEmbeddingComposite

    sampler = DWaveSampler(solver=solver_id) if solver_id else qa_solver.get_sampler(config["qa"])
    composite = FixedEmbeddingComposite(sampler, embedding)

    bqm = sa_solver.to_bqm(model)
    chain_strength = qa_solver.compute_chain_strength(bqm, float(chain_strength_alpha))
    reads = int(num_reads if num_reads is not None else config["qa"]["num_reads"])
    anneal = float(
        annealing_time if annealing_time is not None else config["qa"]["annealing_time"]
    )

    start = time.perf_counter()
    sampleset = composite.sample(
        bqm,
        num_reads=reads,
        annealing_time=anneal,
        chain_strength=chain_strength,
        answer_mode="raw",
    )
    sampleset.resolve()
    elapsed = time.perf_counter() - start

    fields = sampleset.record.dtype.names
    chain_break = (
        float(np.mean(sampleset.record.chain_break_fraction))
        if "chain_break_fraction" in fields else float("nan")
    )
    chain_break_max = (
        float(np.max(sampleset.record.chain_break_fraction))
        if "chain_break_fraction" in fields else float("nan")
    )
    timing = dict(sampleset.info.get("timing", {}))

    return analyze_sampleset(
        model=model,
        instance=instance,
        sampleset=sampleset,
        solver="QA",
        runtime=elapsed,
        num_reads=reads,
        tolerance=float(config["feasibility"]["tolerance"]),
        extra={
            # 이름은 notebook 05의 qa_results.csv 스키마를 따른다.
            "qa_chain_strength_alpha": float(chain_strength_alpha),
            "qa_chain_strength": chain_strength,
            "qa_chain_break_method": str(
                config["qa"].get("chain_break_method", "majority_vote")
            ),
            "qa_chain_break_fraction": chain_break,
            "qa_chain_break_max": chain_break_max,
            "qa_annealing_time": anneal,
            "qa_wall_clock": elapsed,
            "qa_qpu_access_time_us": float(
                timing.get("qpu_access_time", float("nan"))
            ),
            "qa_qpu_sampling_time_us": float(
                timing.get("qpu_sampling_time", float("nan"))
            ),
            "solver_id": str(sampler.solver.id),
        },
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """embedding 품질과 QA 결과를 나란히 놓는다.

    embedding이 좋을수록(큐빗 적고 chain 짧을수록) chain break가 낮고
    feasible 비율이 높은지 확인하기 위한 표이다.
    """
    columns = [
        column for column in (
            "seed", "selected_by", "physical_qubits", "avg_chain_length",
            "max_chain_length", "chain_strength_alpha", "qa_chain_break_fraction",
            "qa_chain_break_max", "feasible_fraction", "best_feasible_objective",
            "true_gap_percent", "status",
        ) if column in results.columns
    ]
    return results[columns].sort_values("physical_qubits")


# notebook 05의 qa_results.csv와 동일한 컬럼 구성. 두 파일을 나중에
# concat 해서 비교할 수 있도록 순서까지 맞춘다.
QA_SCHEMA: tuple[str, ...] = (
    "instance", "size", "formulation", "linking",
    "qubo_variables", "qubo_quadratic_terms",
    "solver", "status", "runtime", "num_reads",
    "best_energy", "best_objective", "best_is_feasible",
    "best_total_violation", "best_max_violation",
    "feasible_fraction", "best_feasible_objective", "best_feasible_energy",
    "num_samples", "energy_mismatch", "reported_best_energy",
    "argmin_agreement", "unique_samples", "feasible_reads",
    "qa_chain_strength", "qa_chain_strength_alpha", "qa_annealing_time",
    "qa_chain_break_method", "qa_chain_break_fraction",
    "qa_wall_clock", "qa_qpu_access_time_us", "qa_qpu_sampling_time_us",
    "embedding_status", "logical_variables", "physical_qubits",
    "max_chain_length", "mean_chain_length", "embedding_search_time",
    "qa_message",
)

# 스키마에는 없지만 이 실험에만 필요한 컬럼. seed가 없으면 같은 instance의
# 여러 embedding을 구분할 수 없으므로 뒤에 덧붙인다.
EXTRA_COLUMNS: tuple[str, ...] = ("seed", "selected_by", "solver_id")


def to_schema_row(
    outcome: SolverOutcome,
    meta: dict[str, Any],
    model: QUBOModel,
    instance: CFLPInstance,
    formulation: str,
    selected_by: str = "",
    include_linking: bool = False,
) -> dict[str, Any]:
    """QA 결과를 qa_results.csv와 동일한 컬럼 구성으로 정리한다.

    embedding 탐색 정보(meta)와 QUBO 정보(model)를 합쳐 한 행으로 만든다.
    스키마에 없는 값은 빠지고, 스키마에 있으나 이 실험에서 생기지 않는 값은
    NaN 또는 빈 문자열이 된다.
    """
    source = outcome.to_record()
    source.update({
        "instance": instance.name,
        "size": instance.num_customers,
        "formulation": formulation,
        "linking": bool(include_linking),
        "qubo_variables": model.num_variables,
        "qubo_quadratic_terms": model.num_quadratic_terms,
        "embedding_status": "OK",
        "logical_variables": meta.get("logical_variables", model.num_variables),
        "physical_qubits": meta.get("physical_qubits", np.nan),
        "max_chain_length": meta.get("max_chain_length", np.nan),
        # qa_results.csv는 평균 chain을 mean_chain_length로 부른다.
        "mean_chain_length": meta.get("avg_chain_length", np.nan),
        "embedding_search_time": meta.get("embedding_search_time", np.nan),
        "qa_message": "",
        "seed": meta.get("seed"),
        "selected_by": selected_by,
    })
    row = {column: source.get(column, np.nan) for column in QA_SCHEMA}
    for column in EXTRA_COLUMNS:
        row[column] = source.get(column, np.nan)
    return row


def to_schema_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """행 목록을 스키마 순서의 DataFrame으로 만든다."""
    frame = pd.DataFrame(rows)
    ordered = [c for c in QA_SCHEMA if c in frame.columns]
    ordered += [c for c in EXTRA_COLUMNS if c in frame.columns]
    return frame[ordered]


# 위반량 히스토그램의 구간. 등식 제약이라 0만이 feasible이므로 0을 따로 세고,
# 그 바로 위 구간을 촘촘히 나눈다. "거의 다 왔는데 못 맞춘 것"인지
# "한참 먼 것"인지를 구분하기 위해서다.
VIOLATION_BINS: tuple[float, ...] = (0, 1, 2, 3, 5, 10, 20, 50, 100, np.inf)


def violation_distribution(
    model: QUBOModel,
    instance: CFLPInstance,
    sampleset: Any,
    tolerance: float,
    bins: Sequence[float] = VIOLATION_BINS,
) -> pd.DataFrame:
    """sample들의 총 제약 위반량 분포를 구한다.

    best sample 하나만 보면 "최소 위반 26"이라는 숫자밖에 남지 않는다.
    전체 분포를 보면 26이 외톨이인지, 아니면 낮은 위반이 두텁게 쌓여 있는데
    0만 못 찍는 것인지 알 수 있다. 후자라면 정밀도를 조금만 올려도 feasible이
    나올 수 있다는 뜻이고, 전자라면 구조적으로 멀다는 뜻이다.

    구간은 (이전 경계, 현재 경계] 형태이며, 위반 0은 feasible이므로 따로 센다.
    """
    violations, weights = sample_violations(
        model, instance, sampleset, tolerance
    )
    total = weights.sum()

    rows: list[dict[str, Any]] = [{
        "range": "0 (feasible)",
        "count": int(weights[violations <= tolerance].sum()),
    }]
    lower = tolerance
    for upper in bins[1:]:
        if np.isfinite(upper):
            mask = (violations > lower) & (violations <= upper)
            label = f"({0 if lower == tolerance else lower:g}, {upper:g}]"
        else:
            mask = violations > lower
            label = f"> {lower:g}"
        rows.append({"range": label, "count": int(weights[mask].sum())})
        lower = upper

    frame = pd.DataFrame(rows)
    frame["fraction"] = frame["count"] / total if total else 0.0
    return frame


def sample_violations(
    model: QUBOModel,
    instance: CFLPInstance,
    sampleset: Any,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """각 sample의 총 위반량과 등장 횟수를 반환한다."""
    from .evaluation import _sampleset_to_arrays

    samples, energies, occurrences = _sampleset_to_arrays(model, sampleset)
    violations = np.array([
        evaluate_sample(model, instance, samples[row], tolerance,
                        energy=energies[row]).total_violation
        for row in range(samples.shape[0])
    ])
    return violations, np.asarray(occurrences, dtype=float)


def violation_summary(
    model: QUBOModel,
    instance: CFLPInstance,
    sampleset: Any,
    tolerance: float,
) -> dict[str, Any]:
    """위반량의 요약 통계. 최소값이 얼마나 외톨이인지 본다."""
    violations, _ = sample_violations(model, instance, sampleset, tolerance)
    return {
        "min_violation": float(violations.min()),
        "p01_violation": float(np.percentile(violations, 1)),
        "p05_violation": float(np.percentile(violations, 5)),
        "median_violation": float(np.median(violations)),
        "max_violation": float(violations.max()),
        "num_violation_le_1": int((violations <= 1).sum()),
        "num_violation_le_3": int((violations <= 3).sum()),
        "num_violation_le_5": int((violations <= 5).sum()),
        "num_violation_le_10": int((violations <= 10).sum()),
    }
