"""MS formulation의 embedding 실패 원인을 분석하는 실험 모듈.

본 실험(notebook 05)에서 MS는 6x6 이상에서 embedding에 실패했다. 원인 후보가
세 가지인데, 관측만으로는 구분되지 않는다.

    (a) search budget  : timeout/tries가 부족했다
    (b) topology       : Pegasus의 connectivity가 부족하다
    (c) formulation    : MS QUBO 그래프 자체가 너무 조밀하다

이 모듈은 셋을 분리하기 위해 timeout, tries, topology를 축으로 sweep한다.

판정 규칙
--------
    timeout/tries를 늘렸더니 성공        -> (a) search budget
    Pegasus 실패, Zephyr 성공            -> (b) topology
    충분한 budget + Zephyr에서도 실패    -> (c) formulation 자체

주의: embedding 탐색은 확률적이므로 seed 하나로 판정하면 안 된다. 각 조건마다
여러 seed를 돌려 **성공률**로 본다.

실행 시간
--------
실패하는 경우는 timeout을 **전부 소진**한다. 따라서 최악의 경우 소요 시간은
    sum(timeouts) x len(seeds) x len(topologies) x len(targets)
가 되며, 큰 instance에서는 수 시간이 걸릴 수 있다. ``estimate_budget``으로
먼저 확인한 뒤 실행할 것.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .data_generator import CFLPInstance
from .qubo_builder import build_qubo
from .sa_solver import to_bqm

# 기본 topology 크기.
#   pegasus 16 : Advantage 세대 (5,640 노드, 평균차수 14.4)
#   zephyr  15 : Advantage2 세대의 이상적 전체 크기 (7,440 노드, 평균차수 19.3)
# 둘 다 결함 없는 이상적 그래프이므로 실제 QPU보다 낙관적이다.
DEFAULT_TOPOLOGIES: tuple[tuple[str, int], ...] = (("pegasus", 16), ("zephyr", 15))

_TARGET_CACHE: dict[tuple[str, int], list[tuple[int, int]]] = {}

# topology 이름만 준 경우에 쓰는 기본 크기.
DEFAULT_TOPOLOGY_SIZE: dict[str, int] = {"pegasus": 16, "zephyr": 15}


def normalize_topologies(
    topologies: Any,
) -> list[tuple[str, int]]:
    """topology 인자를 [(이름, 크기), ...] 형태로 정규화한다.

    다음을 모두 허용한다.

        "zephyr"                     -> [("zephyr", 15)]
        ["zephyr", "pegasus"]        -> [("zephyr", 15), ("pegasus", 16)]
        ("zephyr", 15)               -> [("zephyr", 15)]
        [("zephyr", 15)]             -> [("zephyr", 15)]

    크기를 생략하면 ``DEFAULT_TOPOLOGY_SIZE``를 쓴다.
    """
    def one(item: Any) -> tuple[str, int]:
        if isinstance(item, str):
            name = item.lower()
            if name not in DEFAULT_TOPOLOGY_SIZE:
                raise ValueError(
                    f"지원하지 않는 topology입니다: {item}. "
                    f"사용 가능: {sorted(DEFAULT_TOPOLOGY_SIZE)}"
                )
            return name, DEFAULT_TOPOLOGY_SIZE[name]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return str(item[0]).lower(), int(item[1])
        raise ValueError(
            f"topology 항목의 형식이 올바르지 않습니다: {item!r}. "
            f'"zephyr" 또는 ("zephyr", 15) 형태여야 합니다.'
        )

    # 단일 문자열
    if isinstance(topologies, str):
        return [one(topologies)]
    # ("zephyr", 15) 처럼 이름+크기 한 쌍
    if (
        isinstance(topologies, (list, tuple))
        and len(topologies) == 2
        and isinstance(topologies[0], str)
        and isinstance(topologies[1], int)
    ):
        return [one(topologies)]
    if isinstance(topologies, (list, tuple)):
        return [one(item) for item in topologies]
    raise ValueError(f"topology 인자를 해석할 수 없습니다: {topologies!r}")


def target_edges(topology: str, size: int) -> list[tuple[int, int]]:
    """topology 그래프의 엣지 목록을 만든다 (캐시).

    Args:
        topology: ``"pegasus"`` 또는 ``"zephyr"``.
        size: 그래프 크기 파라미터 (P16, Z15 등의 숫자).
    """
    key = (topology.lower(), int(size))
    if key in _TARGET_CACHE:
        return _TARGET_CACHE[key]

    import dwave_networkx as dnx

    builders = {"pegasus": dnx.pegasus_graph, "zephyr": dnx.zephyr_graph}
    if key[0] not in builders:
        raise ValueError(
            f"지원하지 않는 topology입니다: {topology}. "
            f"사용 가능: {sorted(builders)}"
        )
    edges = list(builders[key[0]](key[1]).edges())
    _TARGET_CACHE[key] = edges
    return edges


def list_available_solvers() -> pd.DataFrame:
    """계정에서 쓸 수 있는 QPU solver와 topology를 조회한다.

    online 모드에서 어떤 topology를 지정할 수 있는지 먼저 확인할 때 쓴다.
    """
    from dwave.cloud import Client

    rows: list[dict[str, Any]] = []
    with Client.from_config() as client:
        for solver in client.get_solvers(qpu=True):
            topology = solver.properties.get("topology", {})
            rows.append({
                "solver": solver.id,
                "topology": topology.get("type", "unknown"),
                "topology_shape": str(topology.get("shape", "")),
                "qubits": len(solver.nodes),
                "couplers": len(solver.edges),
            })
    return pd.DataFrame(rows)


def qpu_target_edges(
    qa_config: dict[str, Any], topology: str | None = None
) -> tuple[list[tuple[int, int]], str, str]:
    """실제 QPU의 working graph를 가져온다 (online 모드).

    Args:
        qa_config: 설정의 ``qa`` 섹션. ``solver_name``이 있으면 그것을 우선한다.
        topology: ``"pegasus"`` 또는 ``"zephyr"``. 지정하면 해당 topology를
            가진 solver를 골라 접속한다. None이면 기본 solver를 쓴다.

    Returns:
        (엣지 목록, topology 이름, solver 이름).

    Raises:
        RuntimeError: 요청한 topology의 solver가 계정에 없는 경우.
            사용 가능한 solver 목록을 함께 알려준다.

    결함 큐빗이 제외된 실제 그래프이므로 이상적 그래프보다 불리하다.
    """
    from dwave.system import DWaveSampler

    solver_name = qa_config.get("solver_name")
    try:
        if solver_name:
            sampler = DWaveSampler(solver=solver_name)
        elif topology:
            sampler = DWaveSampler(solver={"topology__type": topology.lower()})
        else:
            sampler = DWaveSampler()
    except Exception as error:  # noqa: BLE001
        if topology:
            try:
                available = list_available_solvers()
                hint = (
                    "\n계정에서 사용 가능한 QPU solver:\n"
                    + available.to_string(index=False)
                )
            except Exception:  # noqa: BLE001
                hint = ""
            raise RuntimeError(
                f"topology '{topology}'를 가진 QPU solver에 접속하지 못했습니다: "
                f"{error}. Zephyr는 Advantage2 계열이므로 계정에 해당 solver가 "
                f"없으면 사용할 수 없습니다. qa_dryrun=True로 두면 이상적 "
                f"그래프로 비교할 수 있습니다.{hint}"
            ) from error
        raise

    name = str(sampler.properties.get("topology", {}).get("type", "unknown"))
    if topology and name.lower() != topology.lower():
        raise RuntimeError(
            f"요청한 topology '{topology}'와 접속된 solver의 topology "
            f"'{name}'가 다릅니다. config의 qa.solver_name이 설정되어 있으면 "
            f"그것이 우선하므로 확인하십시오."
        )
    return list(sampler.edgelist), name, str(sampler.solver.id)


def _source_graph(
    instance: CFLPInstance, formulation: str, config: dict[str, Any]
) -> tuple[list[tuple[Any, Any]], int, int]:
    """QUBO 그래프의 엣지 목록과 논리 변수/엣지 수를 반환한다.

    embedding 가능 여부는 penalty 값과 무관하고 **그래프 구조에만** 의존하므로
    본 실험의 기본 설정으로 한 번만 만든다.
    """
    model = build_qubo(
        instance,
        formulation,
        float(config["penalty"]["margin"]),
        int(config["encoding"]["precision"]),
    )
    bqm = to_bqm(model)
    edges = [(u, v) for u, v in bqm.quadratic]
    return edges, model.num_variables, len(edges)


def graph_fingerprint(edges: Sequence[tuple[Any, Any]]) -> str:
    """엣지 집합의 지문. 두 그래프가 동일한지 한 값으로 비교한다.

    QPU working graph는 교정 때마다 바뀔 수 있고 D-Wave는 이를 graph_id로
    추적한다. 그런데 graph_id는 지정할 수 없으므로, 저장한 embedding이
    지금 그래프에서도 유효한지 판단하려면 그래프 자체의 지문이 필요하다.
    """
    normalized = sorted(tuple(sorted(map(str, edge))) for edge in edges)
    return hashlib.sha256(repr(normalized).encode()).hexdigest()[:16]


def save_embedding(
    embedding: dict[Any, list[int]],
    directory: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """embedding mapping을 JSON으로 저장한다.

    통계만으로는 재사용할 수 없다. 실제 QA를 돌리려면 FixedEmbeddingComposite에
    넘길 mapping 자체가 필요하고, chain strength 비교나 chain break 측정도
    같은 mapping을 고정해야 의미가 있다.

    함께 저장하는 것:
        - source/target 그래프 지문 : 나중에 유효성 검사에 쓴다
        - solver_id, 탐색 조건       : 어떤 조건에서 찾았는지 추적
        - minorminer/ocean 버전      : 재현 환경 기록
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"{metadata.get('instance', 'unknown')}_"
        f"{metadata.get('formulation', 'NA')}_"
        f"{metadata.get('topology_label', 'NA').replace(':', '-')}_"
        f"seed{metadata.get('seed', 0)}_tries{metadata.get('tries', 0)}.json"
    )
    payload = dict(metadata)
    payload["embedding"] = {str(key): list(map(int, chain))
                            for key, chain in embedding.items()}
    try:
        import minorminer
        payload["minorminer_version"] = getattr(minorminer, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        payload["minorminer_version"] = "unknown"

    path = directory / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_embedding(path: str | Path) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """저장된 embedding과 메타데이터를 읽는다."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    embedding = payload.pop("embedding")
    return embedding, payload


def validate_embedding(
    embedding: dict[Any, list[int]],
    source: Sequence[tuple[Any, Any]],
    target: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """저장된 embedding이 주어진 그래프에서 여전히 유효한지 검사한다.

    working graph가 바뀌면 (1) chain에 쓰인 물리 큐빗이 사라지거나
    (2) chain 내부 연결이 끊기거나 (3) 논리 엣지에 대응하는 물리 연결이
    사라질 수 있다. 이 경우 mapping은 **재확인이 아니라 재탐색이 필요**하다.

    Returns:
        각 조건의 통과 여부와 문제가 된 항목.
    """
    target_nodes = {node for edge in target for node in edge}
    target_set = {tuple(sorted(edge)) for edge in target}

    missing_qubits = sorted(
        {qubit for chain in embedding.values() for qubit in chain}
        - target_nodes
    )
    broken_chains = [
        str(key) for key, chain in embedding.items()
        if len(chain) > 1 and not _chain_connected(chain, target_set)
    ]
    missing_links = [
        (str(u), str(v)) for u, v in source
        if not _has_link(embedding, u, v, target_set)
    ]
    return {
        "valid": not (missing_qubits or broken_chains or missing_links),
        "missing_qubits": missing_qubits[:10],
        "num_missing_qubits": len(missing_qubits),
        "broken_chains": broken_chains[:10],
        "num_broken_chains": len(broken_chains),
        "missing_links": missing_links[:10],
        "num_missing_links": len(missing_links),
    }


def _chain_connected(chain: Sequence[int], target_set: set) -> bool:
    """chain 내부가 target 그래프에서 연결되어 있는지 확인한다."""
    remaining = set(chain)
    frontier = {next(iter(remaining))}
    seen: set[int] = set()
    while frontier:
        node = frontier.pop()
        seen.add(node)
        for other in remaining - seen:
            if tuple(sorted((node, other))) in target_set:
                frontier.add(other)
    return seen == remaining


def _has_link(
    embedding: dict[Any, list[int]], u: Any, v: Any, target_set: set
) -> bool:
    """두 논리변수의 chain 사이에 물리 연결이 하나라도 있는지 확인한다."""
    chain_u, chain_v = embedding.get(u), embedding.get(v)
    if not chain_u or not chain_v:
        return False
    return any(
        tuple(sorted((a, b))) in target_set for a in chain_u for b in chain_v
    )


def run_trial(
    source: Sequence[tuple[Any, Any]],
    target: Sequence[tuple[int, int]],
    timeout: int,
    tries: int,
    seed: int,
    return_embedding: bool = False,
) -> dict[str, Any]:
    """embedding을 한 번 시도하고 결과와 품질 지표를 반환한다.

    Args:
        return_embedding: True면 결과에 ``"embedding"`` 키로 mapping 자체를
            포함한다. 통계만으로는 QA 재실행이나 chain break 비교를 할 수
            없으므로, 성공한 embedding은 반드시 저장해 두어야 한다.
    """
    import minorminer

    start = time.perf_counter()
    embedding = minorminer.find_embedding(
        list(source), list(target), random_seed=int(seed),
        tries=int(tries), timeout=int(timeout),
    )
    elapsed = time.perf_counter() - start

    if not embedding:
        return {
            "success": False,
            "elapsed": elapsed,
            "physical_qubits": np.nan,
            "avg_chain_length": np.nan,
            "max_chain_length": np.nan,
        }
    chains = [len(chain) for chain in embedding.values()]
    result = {
        "success": True,
        "elapsed": elapsed,
        "physical_qubits": int(sum(chains)),
        "avg_chain_length": float(np.mean(chains)),
        "max_chain_length": int(max(chains)),
    }
    if return_embedding:
        result["embedding"] = embedding
    return result


def estimate_budget(
    targets: Sequence[tuple[str, str]],
    topologies: Sequence[tuple[str, int]],
    timeouts: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, float]:
    """최악의 경우(전부 실패) 소요 시간을 추정한다.

    실패하면 timeout을 전부 쓰므로 이 값이 상한이다. 성공하면 훨씬 짧다.
    """
    runs = len(targets) * len(topologies) * len(timeouts) * len(seeds)
    worst = (
        float(sum(timeouts)) * len(seeds) * len(topologies) * len(targets)
    )
    return {
        "runs": runs,
        "worst_case_seconds": worst,
        "worst_case_minutes": worst / 60.0,
        "worst_case_hours": worst / 3600.0,
    }


def run_timeout_sweep(
    targets: Sequence[tuple[str, str]],
    config: dict[str, Any],
    data_dir: Any,
    timeouts: Sequence[int],
    seeds: Sequence[int],
    tries: int,
    topologies: Sequence[tuple[str, int]] = DEFAULT_TOPOLOGIES,
    qa_dryrun: bool = True,
    verbose: bool = True,
    embedding_dir: str | Path | None = None,
) -> pd.DataFrame:
    """timeout을 축으로 sweep한다.

    Args:
        targets: (instance 이름, formulation) 목록.
        timeouts: 시도할 timeout(초) 목록.
        seeds: 각 조건에서 반복할 random seed 목록.
        tries: timeout 실험 동안 **고정**하는 tries 값.
        topologies: (topology 이름, 크기) 목록. dry-run에서만 쓰인다.
        qa_dryrun: True면 이상적 그래프, False면 실제 QPU working graph.
    """
    if qa_dryrun:
        target_list = [
            (f"{name}{size}", name, size, target_edges(name, size))
            for name, size in normalize_topologies(topologies)
        ]
    else:
        # online 모드에서도 topology를 지정할 수 있다. 계정에 해당 topology의
        # solver가 있으면 그것을 골라 접속한다 (Zephyr는 Advantage2 계열).
        target_list = []
        for name, _size in normalize_topologies(topologies):
            edges, actual, solver_id = qpu_target_edges(config["qa"], topology=name)
            print(f"[online] topology '{actual}' solver '{solver_id}' 사용")
            target_list.append((f"qpu:{solver_id}", actual, -1, edges))

    records: list[dict[str, Any]] = []
    for instance_name, formulation in targets:
        instance = CFLPInstance.load(f"{data_dir}/{instance_name}.json")
        source, num_variables, num_edges = _source_graph(instance, formulation, config)
        source_hash = graph_fingerprint(source)
        for label, topology, size, edges in target_list:
            target_hash = graph_fingerprint(edges)
            for timeout in timeouts:
                for seed in seeds:
                    result = run_trial(
                        source, edges, timeout, tries, seed,
                        return_embedding=embedding_dir is not None,
                    )
                    record = {
                        "instance": instance_name,
                        "formulation": formulation,
                        "topology": topology,
                        "topology_label": label,
                        "topology_size": size,
                        "timeout": int(timeout),
                        "tries": int(tries),
                        "seed": int(seed),
                        "logical_variables": num_variables,
                        "logical_edges": num_edges,
                        "sweep": "timeout",
                    }
                    embedding = result.pop("embedding", None)
                    record.update(result)
                    record["source_hash"] = source_hash
                    record["target_hash"] = target_hash
                    record["embedding_overhead"] = (
                        record["physical_qubits"] / num_variables
                        if record["success"]
                        else np.nan
                    )
                    # 성공한 mapping은 반드시 파일로 남긴다. 통계만으로는
                    # QA 재실행도, chain break 비교도 할 수 없다.
                    if embedding and embedding_dir is not None:
                        path = save_embedding(embedding, embedding_dir, record)
                        record["embedding_file"] = path.name
                    records.append(record)
                    if verbose:
                        mark = "성공" if result["success"] else "실패"
                        print(
                            f"  {instance_name} {formulation} {label} "
                            f"timeout={timeout:4d}s seed={seed:<5d} {mark} "
                            f"({result['elapsed']:6.1f}s)",
                            flush=True,
                        )
    return pd.DataFrame(records)


def run_tries_sweep(
    targets: Sequence[tuple[str, str]],
    config: dict[str, Any],
    data_dir: Any,
    tries_grid: Sequence[int],
    seeds: Sequence[int],
    timeout: int,
    topologies: Sequence[tuple[str, int]] = DEFAULT_TOPOLOGIES,
    qa_dryrun: bool = True,
    verbose: bool = True,
    embedding_dir: str | Path | None = None,
) -> pd.DataFrame:
    """timeout을 고정하고 tries를 축으로 sweep한다.

    Args:
        tries_grid: 시도할 tries 값 목록.
        timeout: **고정**하는 timeout(초). timeout sweep 결과를 보고 정한다.
    """
    if qa_dryrun:
        target_list = [
            (f"{name}{size}", name, size, target_edges(name, size))
            for name, size in normalize_topologies(topologies)
        ]
    else:
        # online 모드에서도 topology를 지정할 수 있다. 계정에 해당 topology의
        # solver가 있으면 그것을 골라 접속한다 (Zephyr는 Advantage2 계열).
        target_list = []
        for name, _size in normalize_topologies(topologies):
            edges, actual, solver_id = qpu_target_edges(config["qa"], topology=name)
            print(f"[online] topology '{actual}' solver '{solver_id}' 사용")
            target_list.append((f"qpu:{solver_id}", actual, -1, edges))

    records: list[dict[str, Any]] = []
    for instance_name, formulation in targets:
        instance = CFLPInstance.load(f"{data_dir}/{instance_name}.json")
        source, num_variables, num_edges = _source_graph(instance, formulation, config)
        source_hash = graph_fingerprint(source)
        for label, topology, size, edges in target_list:
            target_hash = graph_fingerprint(edges)
            for tries in tries_grid:
                for seed in seeds:
                    result = run_trial(
                        source, edges, timeout, tries, seed,
                        return_embedding=embedding_dir is not None,
                    )
                    record = {
                        "instance": instance_name,
                        "formulation": formulation,
                        "topology": topology,
                        "topology_label": label,
                        "topology_size": size,
                        "timeout": int(timeout),
                        "tries": int(tries),
                        "seed": int(seed),
                        "logical_variables": num_variables,
                        "logical_edges": num_edges,
                        "sweep": "tries",
                    }
                    embedding = result.pop("embedding", None)
                    record.update(result)
                    record["source_hash"] = source_hash
                    record["target_hash"] = target_hash
                    record["embedding_overhead"] = (
                        record["physical_qubits"] / num_variables
                        if record["success"]
                        else np.nan
                    )
                    # 성공한 mapping은 반드시 파일로 남긴다. 통계만으로는
                    # QA 재실행도, chain break 비교도 할 수 없다.
                    if embedding and embedding_dir is not None:
                        path = save_embedding(embedding, embedding_dir, record)
                        record["embedding_file"] = path.name
                    records.append(record)
                    if verbose:
                        mark = "성공" if result["success"] else "실패"
                        print(
                            f"  {instance_name} {formulation} {label} "
                            f"tries={tries:3d} seed={seed:<5d} {mark} "
                            f"({result['elapsed']:6.1f}s)",
                            flush=True,
                        )
    return pd.DataFrame(records)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """instance x topology별 요약표를 만든다.

    ``min_timeout_success``는 embedding이 처음 성공한 최소 timeout이다.
    이 값이 존재하면 search budget 문제(원인 a)를 시사하고, 모든 timeout에서
    실패하면 topology 또는 formulation 문제를 시사한다.
    """
    rows: list[dict[str, Any]] = []
    for (instance, formulation, topology), group in results.groupby(
        ["instance", "formulation", "topology"], sort=False
    ):
        successes = group[group["success"]]
        timeout_group = group[group["sweep"] == "timeout"]
        timeout_success = timeout_group[timeout_group["success"]]
        rows.append({
            "instance": instance,
            "formulation": formulation,
            "topology": topology,
            "logical_variables": int(group["logical_variables"].iloc[0]),
            "logical_edges": int(group["logical_edges"].iloc[0]),
            "trials": int(len(group)),
            "success_rate": float(group["success"].mean()),
            "min_timeout_success": (
                int(timeout_success["timeout"].min())
                if not timeout_success.empty
                else np.nan
            ),
            "physical_qubits_median": (
                float(successes["physical_qubits"].median())
                if not successes.empty else np.nan
            ),
            "avg_chain_length": (
                float(successes["avg_chain_length"].mean())
                if not successes.empty else np.nan
            ),
            "max_chain_length": (
                float(successes["max_chain_length"].max())
                if not successes.empty else np.nan
            ),
            "embedding_overhead": (
                float(successes["embedding_overhead"].median())
                if not successes.empty else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["formulation", "logical_variables", "topology"]
    )


def diagnose(results: pd.DataFrame) -> list[str]:
    """세 원인 중 어느 것인지 instance별로 판정한다."""
    lines: list[str] = []
    summary = summarize(results)
    for (instance, formulation), group in summary.groupby(
        ["instance", "formulation"], sort=False
    ):
        rates = dict(zip(group["topology"], group["success_rate"]))
        pegasus = rates.get("pegasus", np.nan)
        zephyr = rates.get("zephyr", np.nan)
        subset = results[
            (results["instance"] == instance)
            & (results["formulation"] == formulation)
        ]
        budget_helped = False
        for topology in ("pegasus", "zephyr"):
            part = subset[
                (subset["topology"] == topology) & (subset["sweep"] == "timeout")
            ]
            if part.empty:
                continue
            by_timeout = part.groupby("timeout")["success"].mean()
            if len(by_timeout) > 1 and by_timeout.iloc[-1] > by_timeout.iloc[0] + 1e-9:
                budget_helped = True

        if budget_helped:
            verdict = "(a) search budget — timeout/tries를 늘리자 성공률이 올랐습니다."
        elif (not np.isnan(zephyr)) and zephyr > 0 and (np.isnan(pegasus) or pegasus == 0):
            verdict = "(b) topology — Pegasus에서는 실패하고 Zephyr에서는 성공합니다."
        elif (np.isnan(pegasus) or pegasus == 0) and (np.isnan(zephyr) or zephyr == 0):
            verdict = "(c) formulation — 충분한 budget과 Zephyr에서도 실패합니다."
        else:
            verdict = "두 topology 모두 성공. 원인 분석 대상이 아닙니다."

        lines.append(
            f"{instance} {formulation} "
            f"(변수 {int(group['logical_variables'].iloc[0])}, "
            f"엣지 {int(group['logical_edges'].iloc[0])}): "
            f"pegasus {pegasus:.0%} / zephyr {zephyr:.0%} -> {verdict}"
        )
    return lines
