"""실험 결과와 해를 파일로 저장/복원하는 모듈.

notebook을 다시 실행하지 않아도 분석이 가능하도록, 각 단계의 결과를
CSV(표)와 NPZ(해 배열)로 남긴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cflp_ms import MSSolution
from .cflp_ss import SSSolution


def save_table(frame: pd.DataFrame, directory: str | Path, filename: str) -> Path:
    """DataFrame을 CSV로 저장한다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    frame.to_csv(path, index=False)
    return path


def load_table(directory: str | Path, filename: str) -> pd.DataFrame:
    """저장된 CSV를 읽어 DataFrame으로 반환한다."""
    path = Path(directory) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"결과 파일이 없습니다: {path}. 앞 단계 notebook을 먼저 실행하십시오."
        )
    return pd.read_csv(path)


def save_solution(
    solution: SSSolution | MSSolution,
    directory: str | Path,
    instance_name: str,
    tag: str,
) -> Path:
    """해를 NPZ로 저장한다.

    Args:
        tag: 해의 종류 식별자 (예: ``"gurobi_SS"``, ``"sa_MS"``).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{instance_name}__{tag}.npz"
    if isinstance(solution, SSSolution):
        np.savez(path, kind="SS", x=solution.x, y=solution.y)
    else:
        np.savez(path, kind="MS", q=solution.q, y=solution.y)
    return path


def load_solution(
    directory: str | Path, instance_name: str, tag: str
) -> SSSolution | MSSolution:
    """저장된 해를 복원한다."""
    path = Path(directory) / f"{instance_name}__{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(f"해 파일이 없습니다: {path}")
    payload = np.load(path, allow_pickle=False)
    kind = str(payload["kind"])
    if kind == "SS":
        return SSSolution(x=payload["x"], y=payload["y"])
    return MSSolution(q=payload["q"], y=payload["y"])


def save_samples(
    samples: dict[str, np.ndarray], directory: str | Path, filename: str
) -> Path:
    """binary sample 배열들을 NPZ로 저장한다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    np.savez_compressed(path, **samples)
    return path


def append_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    """결과 레코드를 목록에 추가한다 (notebook 가독성용 헬퍼)."""
    records.append(dict(record))
