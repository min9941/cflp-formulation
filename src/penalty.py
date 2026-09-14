"""QUBO penalty coefficient(lambda) 계산 모듈.

원칙
----
"constraint violation으로 얻을 수 있는 penalty보다, 원래 objective
function을 통해 얻을 수 있는 최대 개선량이 작아지도록 penalty를 설정한다."

계산식
------
SS와 MS의 목적함수 계수는 모두 비음수이다.

    f_j >= 0,   c_ij d_i >= 0

따라서 어떤 변수 조합을 바꾸더라도 목적함수에서 줄일 수 있는 값의
보수적 상한은 전체 계수의 합이다.

    U_obj = sum_j f_j + sum_i sum_j c_ij d_i

한편 정수 계수를 갖는 등식 제약 g_k(z) = 0이 위반되면 g_k(z)^2 >= 1이므로
penalty 증가분은 최소 lambda이다. 따라서

    lambda = margin * U_obj,   margin > 1

로 두면 어떤 제약 위반도 목적함수 개선으로 상쇄될 수 없다.

주의
----
* lambda를 grid search 하지 않는다.
* instance별 결과를 보고 조정하지 않는다.
* 하나의 formulation 내에서 모든 제약에 동일한 lambda를 사용한다.
* margin은 실험 시작 전에 설정 파일에서 고정한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PenaltyInfo:
    """penalty 계산 결과와 그 근거."""

    objective_upper_bound: float
    margin: float
    value: float

    def to_dict(self) -> dict[str, float]:
        return {
            "penalty_objective_upper_bound": self.objective_upper_bound,
            "penalty_margin": self.margin,
            "penalty_lambda": self.value,
        }


def compute_penalty(objective_upper_bound: float, margin: float) -> PenaltyInfo:
    """목적함수 계수 상한으로부터 penalty coefficient를 계산한다.

    Args:
        objective_upper_bound: 목적함수에서 얻을 수 있는 최대 이득의 상한.
        margin: 안전계수 (1보다 커야 한다).

    Returns:
        ``PenaltyInfo``.

    Raises:
        ValueError: margin이 1 이하이거나 상한이 음수인 경우.
    """
    if margin <= 1.0:
        raise ValueError(f"penalty margin은 1보다 커야 합니다: {margin}")
    if objective_upper_bound < 0:
        raise ValueError(
            f"objective upper bound가 음수입니다: {objective_upper_bound}"
        )
    return PenaltyInfo(
        objective_upper_bound=float(objective_upper_bound),
        margin=float(margin),
        value=float(margin) * float(objective_upper_bound),
    )


def penalty_from_value(value: float, objective_upper_bound: float) -> PenaltyInfo:
    """lambda 값을 직접 지정해 PenaltyInfo를 만든다.

    penalty 민감도 실험(notebook 10) 전용이다. 본 실험의 기본 설정은
    ``compute_penalty``를 사용하며, 이 함수는 lambda를 의도적으로 바꿔가며
    관찰하기 위한 통로이다. margin은 사후적으로 역산해 기록한다.
    """
    if value <= 0:
        raise ValueError(f"penalty lambda는 양수여야 합니다: {value}")
    margin = value / objective_upper_bound if objective_upper_bound else float("nan")
    return PenaltyInfo(
        objective_upper_bound=float(objective_upper_bound),
        margin=float(margin),
        value=float(value),
    )
