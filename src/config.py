"""실험 설정 파일 로딩 모듈.

모든 고정값은 ``config/experiment_config.yaml`` 한 곳에서 관리하며,
코드에서는 이 모듈을 통해서만 접근한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 프로젝트 루트 (src/의 부모 디렉터리)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "experiment_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """YAML 설정 파일을 읽어 dict로 반환한다.

    Args:
        path: 설정 파일 경로. None이면 기본 경로를 사용한다.

    Returns:
        설정 dict.

    Raises:
        FileNotFoundError: 설정 파일이 존재하지 않는 경우.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {config_path}. "
            f"프로젝트 루트에서 실행하고 있는지 확인하십시오."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"설정 파일 형식이 올바르지 않습니다: {config_path}")
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """설정의 ``paths`` 항목을 프로젝트 루트 기준 절대경로로 변환한다.

    Args:
        config: ``load_config``로 읽은 설정 dict.
        key: ``paths`` 하위의 키 이름 (예: ``"raw_dir"``).

    Returns:
        절대경로 ``Path`` 객체. 디렉터리가 없으면 생성한다.
    """
    paths = config.get("paths", {})
    if key not in paths:
        raise KeyError(f"설정의 paths 항목에 '{key}'가 없습니다.")
    resolved = PROJECT_ROOT / paths[key]
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
