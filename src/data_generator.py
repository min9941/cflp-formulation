"""CFLP instance 생성 모듈.

Beasley(OR-Library) 및 Cornuejols-Sridharan-Thizy(1991) 계열의
Capacitated Facility Location Problem instance generation 구조를 따른다.

생성 절차 요약
--------------
1. customer/facility 위치를 단위 정사각형 U[0,1]^2에서 균등 생성
2. 수요 d_i ~ Normal(mu, sigma)를 반올림 후 최소 1로 clip
3. 용량 s_j ~ Uniform[lo, hi]를 생성한 뒤
   sum_j s_j = ratio * sum_i d_i 가 되도록 rescaling (정수)
4. 고정비용 f_j = U[0,90] + U[100,110] * sqrt(s_j)  (용량과 양의 상관)
5. 운송 단가 c_ij = scale * Euclidean distance(i, j)

SS와 MS는 **반드시 동일한 instance data**를 사용한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CFLPInstance:
    """하나의 CFLP instance.

    Attributes:
        name: instance 이름 (예: ``"4x4"``).
        seed: 생성에 사용한 random seed.
        num_customers: 고객 수 |I|.
        num_facilities: 후보 시설 수 |J|.
        demands: 길이 |I|의 정수 수요 배열 d_i.
        capacities: 길이 |J|의 정수 용량 배열 s_j.
        fixed_costs: 길이 |J|의 고정 개설비용 f_j.
        transport_costs: (|I|, |J|) 형태의 단위 운송비 c_ij.
        customer_coords: (|I|, 2) 고객 좌표.
        facility_coords: (|J|, 2) 시설 좌표.
    """

    name: str
    seed: int
    num_customers: int
    num_facilities: int
    demands: np.ndarray
    capacities: np.ndarray
    fixed_costs: np.ndarray
    transport_costs: np.ndarray
    customer_coords: np.ndarray
    facility_coords: np.ndarray

    @property
    def total_demand(self) -> int:
        """전체 수요 합계."""
        return int(self.demands.sum())

    @property
    def total_capacity(self) -> int:
        """전체 용량 합계."""
        return int(self.capacities.sum())

    @property
    def capacity_ratio(self) -> float:
        """전체 용량 / 전체 수요 비율."""
        return self.total_capacity / self.total_demand

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화가 가능한 dict로 변환한다."""
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
        return payload

    def save(self, directory: str | Path) -> Path:
        """instance를 JSON 파일로 저장한다."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CFLPInstance":
        """JSON 파일에서 instance를 복원한다."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"instance 파일이 없습니다: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            name=payload["name"],
            seed=payload["seed"],
            num_customers=payload["num_customers"],
            num_facilities=payload["num_facilities"],
            demands=np.asarray(payload["demands"], dtype=int),
            capacities=np.asarray(payload["capacities"], dtype=int),
            fixed_costs=np.asarray(payload["fixed_costs"], dtype=float),
            transport_costs=np.asarray(payload["transport_costs"], dtype=float),
            customer_coords=np.asarray(payload["customer_coords"], dtype=float),
            facility_coords=np.asarray(payload["facility_coords"], dtype=float),
        )


def _rescale_capacities(
    raw_capacities: np.ndarray, total_demand: int, target_ratio: float
) -> np.ndarray:
    """용량을 목표 비율에 맞게 정수로 rescaling한다.

    단순 반올림은 sum_j s_j / sum_i d_i 비율을 흔들 수 있으므로,
    largest-remainder(최대 잔여) 방식으로 반올림 오차를 배분하여
    목표 총합을 **정확히** 맞춘다.

    Args:
        raw_capacities: rescaling 이전의 실수 용량.
        total_demand: 전체 수요 합계.
        target_ratio: 목표 (전체 용량 / 전체 수요) 비율.

    Returns:
        합계가 정확히 round(target_ratio * total_demand)인 정수 용량 배열.
    """
    target_total = int(round(target_ratio * total_demand))
    scaled = raw_capacities * (target_total / raw_capacities.sum())

    floors = np.floor(scaled).astype(int)
    # 각 시설은 최소 1의 용량을 갖도록 한다.
    floors = np.maximum(floors, 1)

    shortfall = target_total - int(floors.sum())
    if shortfall > 0:
        # 잔여분이 큰 순서대로 1씩 더한다.
        remainders = scaled - np.floor(scaled)
        order = np.argsort(-remainders)
        for k in range(shortfall):
            floors[order[k % len(floors)]] += 1
    elif shortfall < 0:
        # 용량이 큰 순서대로 1씩 뺀다 (최소 1은 유지).
        order = np.argsort(-floors)
        remaining = -shortfall
        idx = 0
        while remaining > 0:
            j = order[idx % len(floors)]
            if floors[j] > 1:
                floors[j] -= 1
                remaining -= 1
            idx += 1
    return floors.astype(int)


def generate_instance(
    name: str, size: int, seed: int, data_config: dict[str, Any]
) -> CFLPInstance:
    """설정에 따라 하나의 CFLP instance를 생성한다.

    Args:
        name: instance 이름.
        size: |I| = |J| = size.
        seed: random seed (재현성 보장).
        data_config: 설정 파일의 ``data`` 섹션.

    Returns:
        생성된 ``CFLPInstance``.
    """
    rng = np.random.default_rng(seed)

    lo = float(data_config["coord_low"])
    hi = float(data_config["coord_high"])
    customer_coords = rng.uniform(lo, hi, size=(size, 2))
    facility_coords = rng.uniform(lo, hi, size=(size, 2))

    # 수요: 정규분포 -> 반올림 -> 최소값 clip
    demands = rng.normal(
        float(data_config["demand_mean"]), float(data_config["demand_std"]), size=size
    )
    demands = np.maximum(
        np.rint(demands).astype(int), int(data_config["demand_min"])
    )
    total_demand = int(demands.sum())

    # 용량: 균등분포 -> 목표 비율에 맞춰 정수 rescaling
    raw_capacities = rng.uniform(
        float(data_config["capacity_low"]),
        float(data_config["capacity_high"]),
        size=size,
    )
    capacities = _rescale_capacities(
        raw_capacities, total_demand, float(data_config["capacity_ratio"])
    )

    # 고정비용: 용량과 양의 상관을 갖도록 sqrt(s_j)에 비례하는 항을 포함
    base = rng.uniform(
        float(data_config["fixed_base_low"]),
        float(data_config["fixed_base_high"]),
        size=size,
    )
    slope = rng.uniform(
        float(data_config["fixed_slope_low"]),
        float(data_config["fixed_slope_high"]),
        size=size,
    )
    fixed_costs = base + slope * np.sqrt(capacities.astype(float))

    # 운송 단가: c_ij = scale * Euclidean distance
    diff = customer_coords[:, None, :] - facility_coords[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=2))
    transport_costs = float(data_config["transportation_scale"]) * distances

    decimals = int(data_config["cost_decimals"])
    transport_costs = np.round(transport_costs, decimals)
    fixed_costs = np.round(fixed_costs, decimals)

    return CFLPInstance(
        name=name,
        seed=seed,
        num_customers=size,
        num_facilities=size,
        demands=demands,
        capacities=capacities,
        fixed_costs=fixed_costs,
        transport_costs=transport_costs,
        customer_coords=customer_coords,
        facility_coords=facility_coords,
    )


def generate_all_instances(config: dict[str, Any]) -> list[CFLPInstance]:
    """설정에 정의된 모든 instance를 생성한다."""
    data_config = config["data"]
    return [
        generate_instance(
            name=spec["name"],
            size=int(spec["size"]),
            seed=int(spec["seed"]),
            data_config=data_config,
        )
        for spec in config["instances"]
    ]


def build_toy_instance(toy_config: dict[str, Any]) -> CFLPInstance:
    """exhaustive validation 전용 toy instance를 만든다.

    실제 실험 instance가 아니며, QUBO 변환의 정확성을 전수 검증하기 위해
    수요/용량을 매우 작게 고정한 별도 instance이다.
    """
    demands = np.asarray(toy_config["demands"], dtype=int)
    capacities = np.asarray(toy_config["capacities"], dtype=int)
    fixed_costs = np.asarray(toy_config["fixed_costs"], dtype=float)
    transport_costs = np.asarray(toy_config["transport_costs"], dtype=float)

    num_customers = int(toy_config["num_customers"])
    num_facilities = int(toy_config["num_facilities"])

    if demands.shape != (num_customers,):
        raise ValueError("toy instance의 demands 길이가 num_customers와 다릅니다.")
    if capacities.shape != (num_facilities,):
        raise ValueError("toy instance의 capacities 길이가 num_facilities와 다릅니다.")
    if transport_costs.shape != (num_customers, num_facilities):
        raise ValueError("toy instance의 transport_costs 형태가 올바르지 않습니다.")

    return CFLPInstance(
        name="toy2x2",
        seed=-1,
        num_customers=num_customers,
        num_facilities=num_facilities,
        demands=demands,
        capacities=capacities,
        fixed_costs=fixed_costs,
        transport_costs=transport_costs,
        customer_coords=np.zeros((num_customers, 2)),
        facility_coords=np.zeros((num_facilities, 2)),
    )
