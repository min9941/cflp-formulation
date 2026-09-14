"""QUBO 해를 원래 CFLP 변수 공간으로 되돌리는(decode) 모듈.

QUBO energy를 직접 비교하지 않고, 반드시 원래 CFLP 변수로 decode한 뒤
원래 objective를 계산한다는 것이 본 실험의 핵심 비교 원칙이다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .cflp_ms import MSSolution
from .cflp_ss import SSSolution
from .data_generator import CFLPInstance
from .qubo_builder import QUBOModel


def sample_to_array(
    model: QUBOModel, sample: Sequence[int] | Mapping[str, int] | Mapping[int, int]
) -> np.ndarray:
    """다양한 형태의 sample을 변수 인덱스 순서의 0/1 배열로 정규화한다.

    Args:
        model: 대상 QUBO 모형.
        sample: 리스트/배열, 변수 이름 dict, 또는 인덱스 dict.

    Returns:
        길이 ``model.num_variables``의 정수 배열.
    """
    if isinstance(sample, Mapping):
        keys = list(sample.keys())
        if keys and isinstance(keys[0], str):
            return np.array(
                [int(sample[name]) for name in model.variables], dtype=int
            )
        return np.array(
            [int(sample[index]) for index in range(model.num_variables)], dtype=int
        )
    values = np.asarray(sample, dtype=int)
    if values.size != model.num_variables:
        raise ValueError(
            f"sample 길이가 QUBO 변수 수와 다릅니다: "
            f"{values.size} != {model.num_variables}"
        )
    return values


def decode_ss(model: QUBOModel, sample: Sequence[int] | Mapping) -> SSSolution:
    """SS QUBO 해를 (x, y)로 decode한다."""
    if model.metadata.get("formulation") != "SS":
        raise ValueError("SS QUBO가 아닙니다.")
    values = sample_to_array(model, sample)
    num_customers = model.metadata["num_customers"]
    num_facilities = model.metadata["num_facilities"]
    x_index = model.metadata["x_index"]
    y_index = model.metadata["y_index"]

    x = np.zeros((num_customers, num_facilities), dtype=float)
    for (i, j), index in x_index.items():
        x[i, j] = float(values[index])
    y = np.array(
        [float(values[y_index[j]]) for j in range(num_facilities)], dtype=float
    )
    return SSSolution(x=x, y=y)


def decode_ms(model: QUBOModel, sample: Sequence[int] | Mapping) -> MSSolution:
    """MS QUBO 해를 (q, y)로 decode한다.

    q_ij = sum_k w_k z_ijk 이며, 원래 비율 변수는 x_ij = q_ij / d_i 이다.
    """
    if model.metadata.get("formulation") != "MS":
        raise ValueError("MS QUBO가 아닙니다.")
    values = sample_to_array(model, sample)
    num_customers = model.metadata["num_customers"]
    num_facilities = model.metadata["num_facilities"]
    q_index = model.metadata["q_index"]
    q_encodings = model.metadata["q_encodings"]
    y_index = model.metadata["y_index"]

    q = np.zeros((num_customers, num_facilities), dtype=float)
    for (i, j), indices in q_index.items():
        weights = q_encodings[(i, j)].weights
        q[i, j] = float(sum(w * int(values[idx]) for w, idx in zip(weights, indices)))
    y = np.array(
        [float(values[y_index[j]]) for j in range(num_facilities)], dtype=float
    )
    return MSSolution(q=q, y=y)


def decode(model: QUBOModel, sample: Sequence[int] | Mapping) -> SSSolution | MSSolution:
    """formulation에 맞는 decode 함수를 호출한다."""
    formulation = model.metadata.get("formulation")
    if formulation == "SS":
        return decode_ss(model, sample)
    if formulation == "MS":
        return decode_ms(model, sample)
    raise ValueError(f"알 수 없는 formulation metadata: {formulation}")


def decode_slacks(model: QUBOModel, sample: Sequence[int] | Mapping) -> np.ndarray:
    """capacity 제약의 slack 값을 decode한다 (검증/진단용)."""
    values = sample_to_array(model, sample)
    slack_index = model.metadata["slack_index"]
    slack_encodings = model.metadata["slack_encodings"]
    num_facilities = model.metadata["num_facilities"]
    slacks = np.zeros(num_facilities, dtype=float)
    for j in range(num_facilities):
        weights = slack_encodings[j].weights
        slacks[j] = float(
            sum(w * int(values[idx]) for w, idx in zip(weights, slack_index[j]))
        )
    return slacks


def decode_linking_slacks(
    model: QUBOModel, sample: Sequence[int] | Mapping
) -> dict[tuple[int, int], float]:
    """linking constraint의 slack 값을 decode한다.

    linking을 포함하지 않은 QUBO에서는 빈 dict를 반환한다.
    """
    if not model.metadata.get("include_linking"):
        return {}
    values = sample_to_array(model, sample)
    linking_index = model.metadata["linking_index"]
    linking_encodings = model.metadata["linking_encodings"]
    return {
        key: float(
            sum(
                w * int(values[idx])
                for w, idx in zip(linking_encodings[key].weights, indices)
            )
        )
        for key, indices in linking_index.items()
    }


def constraint_residuals(
    model: QUBOModel, instance: CFLPInstance, sample: Sequence[int] | Mapping
) -> np.ndarray:
    """QUBO에 penalty로 포함된 등식 제약들의 잔차 g_k(z)를 계산한다.

    penalty term은 lambda * sum_k g_k(z)^2 이므로, 이 잔차를 이용하면
    QUBO energy를 독립적으로 재구성하여 변환의 정확성을 검증할 수 있다.
    """
    formulation = model.metadata["formulation"]
    slacks = decode_slacks(model, sample)
    residuals: list[float] = []

    if formulation == "SS":
        solution = decode_ss(model, sample)
        residuals.extend((solution.x.sum(axis=1) - 1.0).tolist())
        capacity_residual = (
            (instance.demands[:, None] * solution.x).sum(axis=0)
            - instance.capacities * solution.y
            + slacks
        )
        residuals.extend(capacity_residual.tolist())
        primal = solution.x
        bound = np.ones(instance.num_customers, dtype=float)
    else:
        solution = decode_ms(model, sample)
        residuals.extend(
            (solution.q.sum(axis=1) - instance.demands.astype(float)).tolist()
        )
        capacity_residual = (
            solution.q.sum(axis=0) - instance.capacities * solution.y + slacks
        )
        residuals.extend(capacity_residual.tolist())
        primal = solution.q
        bound = instance.demands.astype(float)

    # linking constraint를 포함한 경우에만 그 잔차도 더한다.
    #   SS: x_ij - y_j + s = 0
    #   MS: q_ij - d_i y_j + s = 0
    if model.metadata.get("include_linking"):
        linking_slacks = decode_linking_slacks(model, sample)
        for (i, j), slack_value in linking_slacks.items():
            residuals.append(
                float(primal[i, j] - bound[i] * solution.y[j] + slack_value)
            )

    return np.asarray(residuals, dtype=float)


def _encode_linking(
    model: QUBOModel,
    values: np.ndarray,
    primal: np.ndarray,
    y_rounded: np.ndarray,
    ones: bool,
    bounds: np.ndarray | None = None,
) -> None:
    """linking constraint의 slack 비트를 채운다 (검증용).

    slack = bound_i * y_j - primal_ij 이며, feasible 해에서는 항상
    [0, bound_i] 범위의 정수가 된다.
    """
    if not model.metadata.get("include_linking"):
        return
    linking_index = model.metadata["linking_index"]
    linking_encodings = model.metadata["linking_encodings"]
    for (i, j), indices in linking_index.items():
        bound = 1 if ones else int(bounds[i])
        slack_value = bound * int(y_rounded[j]) - int(primal[i, j])
        bits = linking_encodings[(i, j)].encode(slack_value)
        for bit, index in zip(bits, indices):
            values[index] = int(bit)


def encode_ss_solution(
    model: QUBOModel, instance: CFLPInstance, solution: SSSolution
) -> np.ndarray:
    """알려진 SS 해를 QUBO 변수 표현으로 변환한다 (검증용).

    slack 값은 s_j y_j - sum_i d_i x_ij 로 계산하며, 이는 feasible 해에서
    항상 [0, s_j] 범위의 정수가 된다.
    """
    values = np.zeros(model.num_variables, dtype=int)
    x_index = model.metadata["x_index"]
    y_index = model.metadata["y_index"]
    slack_index = model.metadata["slack_index"]
    slack_encodings = model.metadata["slack_encodings"]

    x_rounded = np.rint(solution.x).astype(int)
    y_rounded = np.rint(solution.y).astype(int)

    for (i, j), index in x_index.items():
        values[index] = int(x_rounded[i, j])
    for j, index in y_index.items():
        values[index] = int(y_rounded[j])

    for j in range(instance.num_facilities):
        used = int((instance.demands * x_rounded[:, j]).sum())
        slack_value = int(instance.capacities[j]) * int(y_rounded[j]) - used
        bits = slack_encodings[j].encode(slack_value)
        for bit, index in zip(bits, slack_index[j]):
            values[index] = int(bit)

    _encode_linking(model, values, x_rounded, y_rounded, ones=True)
    return values


def encode_ms_solution(
    model: QUBOModel, instance: CFLPInstance, solution: MSSolution
) -> np.ndarray:
    """알려진 MS 정수해(q)를 QUBO 변수 표현으로 변환한다 (검증용)."""
    values = np.zeros(model.num_variables, dtype=int)
    q_index = model.metadata["q_index"]
    q_encodings = model.metadata["q_encodings"]
    y_index = model.metadata["y_index"]
    slack_index = model.metadata["slack_index"]
    slack_encodings = model.metadata["slack_encodings"]

    q_rounded = np.rint(solution.q).astype(int)
    y_rounded = np.rint(solution.y).astype(int)

    for (i, j), indices in q_index.items():
        bits = q_encodings[(i, j)].encode(int(q_rounded[i, j]))
        for bit, index in zip(bits, indices):
            values[index] = int(bit)
    for j, index in y_index.items():
        values[index] = int(y_rounded[j])

    for j in range(instance.num_facilities):
        used = int(q_rounded[:, j].sum())
        slack_value = int(instance.capacities[j]) * int(y_rounded[j]) - used
        bits = slack_encodings[j].encode(slack_value)
        for bit, index in zip(bits, slack_index[j]):
            values[index] = int(bit)

    _encode_linking(
        model, values, q_rounded, y_rounded, ones=False, bounds=instance.demands
    )
    return values


def encode_solution(
    model: QUBOModel, instance: CFLPInstance, solution: SSSolution | MSSolution
) -> np.ndarray:
    """formulation에 맞는 encode 함수를 호출한다."""
    if isinstance(solution, SSSolution):
        return encode_ss_solution(model, instance, solution)
    return encode_ms_solution(model, instance, solution)
