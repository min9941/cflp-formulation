"""CFLP formulation을 QUBO로 변환하는 모듈.

변환 원칙 (Zhao et al., 2022)
-----------------------------
1. 이미 binary인 변수(y_j, SS의 x_ij)는 그대로 binary 변수로 사용한다.
2. 정수 변수(MS의 q_ij)는 binary expansion으로 표현한다.
3. 부등식 제약은 slack 변수를 추가해 등식 제약으로 바꾼다.
4. slack도 binary expansion으로 표현한다.
5. 모든 등식 제약 g_k(z) = 0은 penalty lambda * g_k(z)^2 로 목적함수에 포함한다.
6. 원래 목적함수는 QUBO의 linear term으로 유지한다.

최종 QUBO
---------
    Q(z) = f_original(z) + lambda * sum_k g_k(z)^2

z_v^2 = z_v (binary)이므로 제곱 전개 시 제곱항은 대각(linear)으로 흡수된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .binary_encoder import BinaryEncoding, build_encoding
from .data_generator import CFLPInstance
from .penalty import PenaltyInfo, compute_penalty, penalty_from_value
from .slack_encoder import SlackSpecification, build_slack

# energy 벡터화 계산 시 한 번에 생성할 중간 배열의 원소 수 상한.
# 큰 MS instance는 항이 10^5개를 넘으므로 청크로 나누지 않으면 메모리가 폭증한다.
_MAX_CHUNK_ELEMENTS: int = 4_000_000


@dataclass
class QUBOModel:
    """이진 이차 모형(QUBO).

    Attributes:
        Q: (u, v) -> 계수. u <= v이며 u == v는 linear term을 의미한다.
        offset: 상수항. QUBO energy = offset + sum Q[u,v] z_u z_v.
        variables: 인덱스 순서대로의 변수 이름.
        penalty: penalty 계산 정보.
        metadata: 디코딩에 필요한 구조 정보.
    """

    Q: dict[tuple[int, int], float]
    offset: float
    variables: list[str]
    penalty: PenaltyInfo
    metadata: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    @property
    def num_terms(self) -> int:
        """0이 아닌 QUBO 계수의 개수."""
        return sum(1 for value in self.Q.values() if value != 0.0)

    @property
    def num_quadratic_terms(self) -> int:
        """비대각(이차) 항의 개수."""
        return sum(1 for (u, v), value in self.Q.items() if u != v and value != 0.0)

    @property
    def num_linear_terms(self) -> int:
        return sum(1 for (u, v), value in self.Q.items() if u == v and value != 0.0)

    def coefficient_stats(self) -> dict[str, float]:
        """QUBO 계수의 최소/최대/범위."""
        values = [value for value in self.Q.values() if value != 0.0]
        if not values:
            return {"qubo_min": 0.0, "qubo_max": 0.0, "qubo_range": 0.0}
        return {
            "qubo_min": float(min(values)),
            "qubo_max": float(max(values)),
            "qubo_range": float(max(values) - min(values)),
        }

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """QUBO 계수를 (u 인덱스, v 인덱스, 계수) 배열로 캐싱한다.

        energy 계산을 벡터화하기 위한 내부 표현이며, 큰 instance에서
        Python 반복문으로 인한 병목을 제거한다.
        """
        cached = self._cache.get("coefficient_arrays")
        if cached is None:
            if self.Q:
                keys = list(self.Q.keys())
                rows = np.array([u for u, _ in keys], dtype=np.int64)
                cols = np.array([v for _, v in keys], dtype=np.int64)
                values = np.array([self.Q[key] for key in keys], dtype=float)
            else:
                rows = np.zeros(0, dtype=np.int64)
                cols = np.zeros(0, dtype=np.int64)
                values = np.zeros(0, dtype=float)
            cached = (rows, cols, values)
            self._cache["coefficient_arrays"] = cached
        return cached

    def energies(self, samples: np.ndarray) -> np.ndarray:
        """여러 sample의 QUBO energy를 한 번에 계산한다.

        Args:
            samples: (num_samples, num_variables) 형태의 0/1 배열.

        Returns:
            길이 num_samples의 energy 배열.
        """
        matrix = np.atleast_2d(np.asarray(samples, dtype=float))
        if matrix.shape[1] != self.num_variables:
            raise ValueError(
                f"sample 열 수가 변수 수와 다릅니다: "
                f"{matrix.shape[1]} != {self.num_variables}"
            )
        rows, cols, values = self._arrays()
        if values.size == 0:
            return np.full(matrix.shape[0], self.offset, dtype=float)
        chunk_rows = max(1, int(_MAX_CHUNK_ELEMENTS // max(values.size, 1)))
        results = np.empty(matrix.shape[0], dtype=float)
        for start in range(0, matrix.shape[0], chunk_rows):
            block = matrix[start : start + chunk_rows]
            products = block[:, rows] * block[:, cols]
            results[start : start + chunk_rows] = products @ values
        return self.offset + results

    def energy(self, sample: Sequence[int] | Mapping[int, int]) -> float:
        """주어진 binary 할당의 QUBO energy를 계산한다."""
        if isinstance(sample, Mapping):
            values = [int(sample[i]) for i in range(self.num_variables)]
        else:
            values = [int(v) for v in sample]
        if len(values) != self.num_variables:
            raise ValueError(
                f"sample 길이가 변수 수와 다릅니다: {len(values)} != {self.num_variables}"
            )
        return float(self.energies(np.asarray(values, dtype=float)[None, :])[0])

    def to_dimod_dict(self) -> dict[tuple[str, str], float]:
        """dimod / neal에 넘길 수 있는 (변수이름, 변수이름) -> 계수 dict."""
        return {
            (self.variables[u], self.variables[v]): coefficient
            for (u, v), coefficient in self.Q.items()
            if coefficient != 0.0
        }

    def size_summary(self) -> dict[str, int]:
        """변수 수 구성 요약."""
        meta = self.metadata
        return {
            "qubo_variables": self.num_variables,
            "qubo_terms": self.num_terms,
            "qubo_quadratic_terms": self.num_quadratic_terms,
            "decision_variables": int(meta.get("num_decision_bits", 0)),
            "encoding_variables": int(meta.get("num_encoding_bits", 0)),
            "slack_variables": int(meta.get("num_slack_bits", 0)),
        }


class _QUBOBuilder:
    """QUBO를 점진적으로 조립하는 내부 헬퍼."""

    def __init__(self) -> None:
        self._variables: list[str] = []
        self._index: dict[str, int] = {}
        self._Q: dict[tuple[int, int], float] = {}
        self._offset: float = 0.0

    def add_variable(self, name: str) -> int:
        """새 binary 변수를 등록하고 인덱스를 반환한다."""
        if name in self._index:
            raise ValueError(f"중복된 QUBO 변수 이름입니다: {name}")
        index = len(self._variables)
        self._variables.append(name)
        self._index[name] = index
        return index

    def add_linear(self, index: int, coefficient: float) -> None:
        """linear term(대각 성분)을 더한다."""
        if coefficient == 0.0:
            return
        key = (index, index)
        self._Q[key] = self._Q.get(key, 0.0) + float(coefficient)

    def add_quadratic(self, u: int, v: int, coefficient: float) -> None:
        """quadratic term을 더한다 (u != v)."""
        if coefficient == 0.0:
            return
        key = (u, v) if u <= v else (v, u)
        self._Q[key] = self._Q.get(key, 0.0) + float(coefficient)

    def add_squared_penalty(
        self,
        terms: Mapping[int, float],
        constant: float,
        weight: float,
    ) -> None:
        """weight * (sum_v a_v z_v + constant)^2 를 QUBO에 더한다.

        z_v^2 = z_v 이므로

            (sum a_v z_v + c)^2
              = sum_v (a_v^2 + 2 a_v c) z_v
              + sum_{u<v} 2 a_u a_v z_u z_v
              + c^2

        Args:
            terms: 변수 인덱스 -> 계수 a_v.
            constant: 상수항 c.
            weight: penalty coefficient lambda.
        """
        items = [(idx, float(a)) for idx, a in terms.items() if a != 0.0]
        for idx, a in items:
            self.add_linear(idx, weight * (a * a + 2.0 * a * constant))
        for position, (idx_u, a_u) in enumerate(items):
            for idx_v, a_v in items[position + 1 :]:
                self.add_quadratic(idx_u, idx_v, weight * 2.0 * a_u * a_v)
        self._offset += weight * constant * constant

    def build(
        self, penalty: PenaltyInfo, metadata: dict[str, Any]
    ) -> QUBOModel:
        """조립 결과를 ``QUBOModel``로 반환한다."""
        cleaned = {key: value for key, value in self._Q.items() if value != 0.0}
        return QUBOModel(
            Q=cleaned,
            offset=self._offset,
            variables=list(self._variables),
            penalty=penalty,
            metadata=metadata,
        )


def _accumulate(target: dict[int, float], index: int, coefficient: float) -> None:
    """같은 변수에 대한 계수를 정확히 합산한다."""
    target[index] = target.get(index, 0.0) + float(coefficient)


def build_ss_qubo(
    instance: CFLPInstance,
    penalty_margin: float,
    lambda_value: float | None = None,
    include_linking: bool = False,
    assignment_multiplier: float = 1.0,
) -> QUBOModel:
    """SS-CFLP를 QUBO로 변환한다.

    assignment_multiplier
        assignment constraint의 penalty에만 곱하는 배수. 기본값 1.0은 모든
        constraint에 동일한 lambda를 쓰는 본 실험의 설정이다.

        capacity constraint는 계수에 d_i, s_j가 들어 있어 제곱 시
        s_j^2 규모로 증폭되는 반면, assignment constraint는 계수가 모두 1이라
        lambda 그대로 남는다. 그 결과 같은 lambda를 써도 하드웨어가 보는
        제약 강도는 s_j^2 배 차이가 난다. 이 배수는 그 불균형을 보정하기
        위한 진단용 손잡이이다.

    변수 구성
        x_ij  : |I| * |J| 개의 원래 binary 변수
        y_j   : |J| 개의 원래 binary 변수
        slack : capacity 제약 |J| 개에 대한 binary expansion

    제약
        (A) sum_j x_ij - 1 = 0
        (C) sum_i d_i x_ij - s_j y_j + slack_j = 0,  slack_j in [0, s_j]
    """
    from .cflp_ss import objective_upper_bound

    num_customers = instance.num_customers
    num_facilities = instance.num_facilities
    demands = instance.demands
    capacities = instance.capacities

    bound = objective_upper_bound(instance)
    penalty = (
        compute_penalty(bound, penalty_margin)
        if lambda_value is None
        else penalty_from_value(lambda_value, bound)
    )
    lam = penalty.value

    builder = _QUBOBuilder()

    x_index: dict[tuple[int, int], int] = {}
    for i in range(num_customers):
        for j in range(num_facilities):
            x_index[(i, j)] = builder.add_variable(f"x_{i}_{j}")

    y_index: dict[int, int] = {}
    for j in range(num_facilities):
        y_index[j] = builder.add_variable(f"y_{j}")

    slack_specs: dict[int, SlackSpecification] = {}
    slack_index: dict[int, list[int]] = {}
    for j in range(num_facilities):
        spec = build_slack(f"cap_{j}", int(capacities[j]))
        slack_specs[j] = spec
        slack_index[j] = [
            builder.add_variable(f"sl_cap_{j}_b{k}") for k in range(spec.num_bits)
        ]

    # ---- linking constraint용 slack (선택) ----
    # x_ij <= y_j 는 capacity constraint에 의해 함의되는 redundant 제약이다.
    # 본 실험의 기본 설정은 이를 제외하지만, 포함했을 때 QUBO가 얼마나
    # 커지는지를 관찰하기 위해 옵션으로 제공한다.
    # x_ij - y_j + s = 0, s in [0, 1] 이므로 slack은 1 bit이다.
    linking_specs: dict[tuple[int, int], SlackSpecification] = {}
    linking_index: dict[tuple[int, int], list[int]] = {}
    if include_linking:
        for i in range(num_customers):
            for j in range(num_facilities):
                spec = build_slack(f"link_{i}_{j}", 1)
                linking_specs[(i, j)] = spec
                linking_index[(i, j)] = [
                    builder.add_variable(f"sl_link_{i}_{j}_b{k}")
                    for k in range(spec.num_bits)
                ]

    # ---- 원래 목적함수 (linear term) ----
    for i in range(num_customers):
        for j in range(num_facilities):
            builder.add_linear(
                x_index[(i, j)],
                float(instance.transport_costs[i, j]) * float(demands[i]),
            )
    for j in range(num_facilities):
        builder.add_linear(y_index[j], float(instance.fixed_costs[j]))

    # ---- (A) single assignment penalty ----
    assignment_weight = lam * float(assignment_multiplier)
    for i in range(num_customers):
        terms: dict[int, float] = {}
        for j in range(num_facilities):
            _accumulate(terms, x_index[(i, j)], 1.0)
        builder.add_squared_penalty(terms, constant=-1.0, weight=assignment_weight)

    # ---- (C) capacity penalty ----
    for j in range(num_facilities):
        terms = {}
        for i in range(num_customers):
            _accumulate(terms, x_index[(i, j)], float(demands[i]))
        _accumulate(terms, y_index[j], -float(capacities[j]))
        for weight, idx in zip(slack_specs[j].encoding.weights, slack_index[j]):
            _accumulate(terms, idx, float(weight))
        builder.add_squared_penalty(terms, constant=0.0, weight=lam)

    # ---- (L) linking penalty (선택) ----
    for (i, j), indices in linking_index.items():
        terms = {}
        _accumulate(terms, x_index[(i, j)], 1.0)
        _accumulate(terms, y_index[j], -1.0)
        for weight, idx in zip(linking_specs[(i, j)].encoding.weights, indices):
            _accumulate(terms, idx, float(weight))
        builder.add_squared_penalty(terms, constant=0.0, weight=lam)

    num_decision_bits = num_customers * num_facilities + num_facilities
    num_slack_bits = sum(spec.num_bits for spec in slack_specs.values()) + sum(
        spec.num_bits for spec in linking_specs.values()
    )

    metadata: dict[str, Any] = {
        "formulation": "SS",
        "instance_name": instance.name,
        "num_customers": num_customers,
        "num_facilities": num_facilities,
        "x_index": x_index,
        "y_index": y_index,
        "slack_index": slack_index,
        "slack_encodings": {j: slack_specs[j].encoding for j in slack_specs},
        "include_linking": include_linking,
        "linking_index": linking_index,
        "linking_encodings": {k: v.encoding for k, v in linking_specs.items()},
        "assignment_multiplier": float(assignment_multiplier),
        "assignment_weight": assignment_weight,
        "num_decision_bits": num_decision_bits,
        "num_encoding_bits": 0,  # SS는 정수 encoding 변수를 사용하지 않는다.
        "num_slack_bits": num_slack_bits,
    }
    return builder.build(penalty, metadata)


def build_ms_qubo(
    instance: CFLPInstance,
    penalty_margin: float,
    precision: int = 1,
    lambda_value: float | None = None,
    include_linking: bool = False,
    assignment_multiplier: float = 1.0,
) -> QUBOModel:
    """MS-CFLP를 QUBO로 변환한다 (1 unit shipment precision).

    assignment_multiplier
        MS에서 SS의 assignment constraint에 대응하는 것은 demand
        satisfaction constraint이다. 이 배수는 그 penalty에만 곱해진다.
        SS와 이름을 맞춘 것은 두 formulation을 같은 축으로 비교하기 위함이다.

    변수 구성
        q_ij  : 각 (i, j)마다 [0, d_i] 범위의 정수를 binary expansion
        y_j   : |J| 개의 원래 binary 변수
        slack : capacity 제약 |J| 개에 대한 binary expansion

    제약
        (D) sum_j q_ij - d_i = 0
        (C) sum_i q_ij - s_j y_j + slack_j = 0,  slack_j in [0, s_j]

    Args:
        precision: unit 정밀도. 본 실험에서는 1만 지원한다.
    """
    from .cflp_ms import objective_upper_bound

    if int(precision) != 1:
        raise NotImplementedError(
            f"본 실험은 1 unit shipment precision만 사용한다 (요청: {precision}). "
            f"다른 정밀도를 쓰려면 encoding 사양을 명시적으로 재정의해야 한다."
        )

    num_customers = instance.num_customers
    num_facilities = instance.num_facilities
    demands = instance.demands
    capacities = instance.capacities

    bound = objective_upper_bound(instance)
    penalty = (
        compute_penalty(bound, penalty_margin)
        if lambda_value is None
        else penalty_from_value(lambda_value, bound)
    )
    lam = penalty.value

    builder = _QUBOBuilder()

    q_encodings: dict[tuple[int, int], BinaryEncoding] = {}
    q_index: dict[tuple[int, int], list[int]] = {}
    for i in range(num_customers):
        encoding = build_encoding(int(demands[i]))
        for j in range(num_facilities):
            q_encodings[(i, j)] = encoding
            q_index[(i, j)] = [
                builder.add_variable(f"q_{i}_{j}_b{k}")
                for k in range(encoding.num_bits)
            ]

    y_index: dict[int, int] = {}
    for j in range(num_facilities):
        y_index[j] = builder.add_variable(f"y_{j}")

    slack_specs: dict[int, SlackSpecification] = {}
    slack_index: dict[int, list[int]] = {}
    for j in range(num_facilities):
        spec = build_slack(f"cap_{j}", int(capacities[j]))
        slack_specs[j] = spec
        slack_index[j] = [
            builder.add_variable(f"sl_cap_{j}_b{k}") for k in range(spec.num_bits)
        ]

    # ---- linking constraint용 slack (선택) ----
    # q_ij <= d_i * y_j 이므로 slack 범위는 [0, d_i]이고,
    # SS의 1 bit와 달리 bit_length(d_i)개가 필요하다. 이 비대칭이
    # linking 제외 여부가 MS에서 훨씬 크게 작용하는 이유이다.
    linking_specs: dict[tuple[int, int], SlackSpecification] = {}
    linking_index: dict[tuple[int, int], list[int]] = {}
    if include_linking:
        for i in range(num_customers):
            for j in range(num_facilities):
                spec = build_slack(f"link_{i}_{j}", int(demands[i]))
                linking_specs[(i, j)] = spec
                linking_index[(i, j)] = [
                    builder.add_variable(f"sl_link_{i}_{j}_b{k}")
                    for k in range(spec.num_bits)
                ]

    # ---- 원래 목적함수 (linear term) ----
    # sum_ij c_ij q_ij = sum_ij c_ij * sum_k w_k z_ijk
    for i in range(num_customers):
        for j in range(num_facilities):
            cost = float(instance.transport_costs[i, j])
            for weight, idx in zip(q_encodings[(i, j)].weights, q_index[(i, j)]):
                builder.add_linear(idx, cost * float(weight))
    for j in range(num_facilities):
        builder.add_linear(y_index[j], float(instance.fixed_costs[j]))

    # ---- (D) demand satisfaction penalty ----
    assignment_weight = lam * float(assignment_multiplier)
    for i in range(num_customers):
        terms: dict[int, float] = {}
        for j in range(num_facilities):
            for weight, idx in zip(q_encodings[(i, j)].weights, q_index[(i, j)]):
                _accumulate(terms, idx, float(weight))
        builder.add_squared_penalty(
            terms, constant=-float(demands[i]), weight=assignment_weight
        )

    # ---- (C) capacity penalty ----
    for j in range(num_facilities):
        terms = {}
        for i in range(num_customers):
            for weight, idx in zip(q_encodings[(i, j)].weights, q_index[(i, j)]):
                _accumulate(terms, idx, float(weight))
        _accumulate(terms, y_index[j], -float(capacities[j]))
        for weight, idx in zip(slack_specs[j].encoding.weights, slack_index[j]):
            _accumulate(terms, idx, float(weight))
        builder.add_squared_penalty(terms, constant=0.0, weight=lam)

    # ---- (L) linking penalty (선택) ----
    for (i, j), indices in linking_index.items():
        terms = {}
        for weight, idx in zip(q_encodings[(i, j)].weights, q_index[(i, j)]):
            _accumulate(terms, idx, float(weight))
        _accumulate(terms, y_index[j], -float(demands[i]))
        for weight, idx in zip(linking_specs[(i, j)].encoding.weights, indices):
            _accumulate(terms, idx, float(weight))
        builder.add_squared_penalty(terms, constant=0.0, weight=lam)

    num_encoding_bits = sum(len(bits) for bits in q_index.values())
    num_slack_bits = sum(spec.num_bits for spec in slack_specs.values()) + sum(
        spec.num_bits for spec in linking_specs.values()
    )

    metadata: dict[str, Any] = {
        "formulation": "MS",
        "instance_name": instance.name,
        "num_customers": num_customers,
        "num_facilities": num_facilities,
        "q_index": q_index,
        "q_encodings": q_encodings,
        "y_index": y_index,
        "slack_index": slack_index,
        "slack_encodings": {j: slack_specs[j].encoding for j in slack_specs},
        "include_linking": include_linking,
        "linking_index": linking_index,
        "linking_encodings": {k: v.encoding for k, v in linking_specs.items()},
        "assignment_multiplier": float(assignment_multiplier),
        "assignment_weight": assignment_weight,
        "num_decision_bits": num_facilities,  # y_j 만이 원래 binary 변수
        "num_encoding_bits": num_encoding_bits,
        "num_slack_bits": num_slack_bits,
    }
    return builder.build(penalty, metadata)


def build_qubo(
    instance: CFLPInstance,
    formulation: str,
    penalty_margin: float,
    precision: int = 1,
    lambda_value: float | None = None,
    include_linking: bool = False,
    assignment_multiplier: float = 1.0,
) -> QUBOModel:
    """formulation 이름으로 QUBO를 생성하는 편의 함수.

    lambda_value를 주면 penalty_margin 대신 그 값을 penalty로 사용한다
    (penalty 민감도 실험 전용).
    """
    key = formulation.upper()
    if key == "SS":
        return build_ss_qubo(
            instance, penalty_margin, lambda_value, include_linking,
            assignment_multiplier,
        )
    if key == "MS":
        return build_ms_qubo(
            instance, penalty_margin, precision, lambda_value, include_linking,
            assignment_multiplier,
        )
    raise ValueError(f"알 수 없는 formulation입니다: {formulation}")


def qubo_statistics(model: QUBOModel) -> dict[str, Any]:
    """결과 저장을 위한 QUBO 통계 dict."""
    stats: dict[str, Any] = {}
    stats.update(model.size_summary())
    stats.update(model.coefficient_stats())
    stats.update(model.penalty.to_dict())
    return stats


def enumerate_all_samples(num_variables: int) -> Iterable[np.ndarray]:
    """모든 binary 할당을 순회한다 (작은 문제 전수 검증 전용)."""
    for code in range(1 << num_variables):
        yield np.array(
            [(code >> position) & 1 for position in range(num_variables)], dtype=int
        )
