"""부등식 제약을 등식 제약으로 바꾸기 위한 slack 변수 encoding.

원칙
----
    a(z) <= b     ==>     a(z) + s = b,   s >= 0

slack s는 Zhao et al. (2022)의 binary expansion으로 표현하되,
**실제로 가능한 slack의 최대값**을 계산하여 필요 이상의 binary 변수를
만들지 않는다.

본 실험의 capacity 제약
-----------------------
    sum_i d_i x_ij - s_j y_j <= 0

좌변의 최솟값은 x = 0, y = 1일 때 -s_j 이므로 slack의 범위는

    s in [0, s_j]

가 된다. 즉 시설 용량 s_j가 그대로 slack의 상한이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .binary_encoder import BinaryEncoding, build_encoding


@dataclass(frozen=True)
class SlackSpecification:
    """하나의 부등식 제약에 대응하는 slack 정보.

    Attributes:
        name: 제약 식별자 (변수 이름 생성에 사용).
        upper_bound: slack이 가질 수 있는 최대값.
        encoding: slack의 binary expansion.
    """

    name: str
    upper_bound: int
    encoding: BinaryEncoding

    @property
    def num_bits(self) -> int:
        return self.encoding.num_bits


def compute_slack_bound(
    coefficients: Sequence[float], rhs: float
) -> int:
    """부등식 ``sum_v a_v z_v <= rhs``의 slack 상한을 계산한다.

    좌변이 binary 변수들의 선형식이므로 최솟값은 음수 계수만 선택했을 때이다.
    slack의 최대값은 ``rhs - min(좌변)``이 된다.

    Args:
        coefficients: 좌변의 계수 a_v.
        rhs: 우변 상수.

    Returns:
        slack 상한 (0 이상의 정수, 내림 처리).

    Raises:
        ValueError: slack 상한이 음수인 경우 (제약이 항상 위반됨).
    """
    minimum_lhs = sum(coefficient for coefficient in coefficients if coefficient < 0)
    bound = rhs - minimum_lhs
    if bound < -1e-9:
        raise ValueError(
            f"slack 상한이 음수입니다 (bound={bound}). 제약이 항상 위반되는 "
            f"구조인지 확인하십시오."
        )
    return int(max(0, int(bound + 1e-9)))


def build_slack(name: str, upper_bound: int) -> SlackSpecification:
    """주어진 상한에 대한 slack 변수 사양을 만든다."""
    return SlackSpecification(
        name=name,
        upper_bound=int(upper_bound),
        encoding=build_encoding(int(upper_bound)),
    )
