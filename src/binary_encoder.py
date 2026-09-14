"""Bounded integer 변수의 binary expansion 모듈.

Zhao et al. (2022)의 binary expansion 방식을 따르되, 표현 범위가
불필요하게 커지지 않도록 **마지막 계수를 잘라내는(truncated) 형태**를 사용한다.

수식
----
상한 U (>= 0)인 정수 변수 q in {0, 1, ..., U}에 대해

    K = bit_length(U)                     (U >= 1인 경우)
    w = [2^0, 2^1, ..., 2^(K-2), U - (2^(K-1) - 1)]
    q = sum_k w_k z_k,   z_k in {0,1}

마지막 계수를 그대로 2^(K-1)로 두면 표현 가능한 최댓값이 2^K - 1이 되어
U를 초과한다. 위와 같이 마지막 계수를 U - (2^(K-1) - 1)로 잘라내면

    sum_k w_k = U

가 되어 표현 범위가 정확히 [0, U]가 된다.

완전성(모든 정수값을 표현 가능한가)
------------------------------------
앞의 K-1개 계수는 0..(2^(K-1) - 1)을 빠짐없이 표현한다. 마지막 계수
w_last = U - 2^(K-1) + 1은 K = bit_length(U)이므로 U <= 2^K - 1에서
w_last <= 2^(K-1)을 만족한다. 즉 마지막 비트를 켜면 표현 구간이
[w_last, U]가 되고, 이는 앞 구간 [0, 2^(K-1) - 1]과 겹치거나 맞닿는다.
따라서 [0, U]의 **모든 정수**를 표현할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryEncoding:
    """상한이 있는 비음 정수 변수의 binary expansion.

    Attributes:
        upper_bound: 표현하려는 정수의 상한 U.
        weights: 각 binary 변수의 계수 w_k. 합은 정확히 U.
    """

    upper_bound: int
    weights: tuple[int, ...]

    @property
    def num_bits(self) -> int:
        """사용하는 binary 변수의 개수."""
        return len(self.weights)

    def decode(self, bits: list[int] | tuple[int, ...]) -> int:
        """binary 값들로부터 정수값을 복원한다."""
        if len(bits) != self.num_bits:
            raise ValueError(
                f"bit 수가 맞지 않습니다: 기대 {self.num_bits}, 입력 {len(bits)}"
            )
        return int(sum(w * int(b) for w, b in zip(self.weights, bits)))

    def encode(self, value: int) -> tuple[int, ...]:
        """정수값을 binary 값들로 변환한다 (greedy, 큰 계수 우선).

        계수가 내림차순으로 정렬되어 있지 않을 수 있으므로 인덱스를
        정렬한 뒤 greedy하게 채운다. 위 완전성 논증에 의해 [0, U] 범위의
        모든 정수에 대해 정확한 표현이 존재한다.
        """
        if not 0 <= value <= self.upper_bound:
            raise ValueError(
                f"값 {value}가 표현 범위 [0, {self.upper_bound}]를 벗어납니다."
            )
        bits = [0] * self.num_bits
        remaining = int(value)
        order = sorted(range(self.num_bits), key=lambda k: -self.weights[k])
        for k in order:
            if self.weights[k] <= remaining:
                bits[k] = 1
                remaining -= self.weights[k]
        if remaining != 0:
            raise RuntimeError(
                f"encoding 실패: 값 {value}를 정확히 표현하지 못했습니다 "
                f"(잔여 {remaining}, weights={self.weights})."
            )
        return tuple(bits)


def build_encoding(upper_bound: int) -> BinaryEncoding:
    """상한 U에 대한 최소 개수의 binary expansion을 생성한다.

    Args:
        upper_bound: 표현하려는 정수의 상한 (0 이상).

    Returns:
        ``BinaryEncoding`` 객체. U = 0이면 binary 변수를 만들지 않는다.

    Raises:
        ValueError: 상한이 음수인 경우.
    """
    upper_bound = int(upper_bound)
    if upper_bound < 0:
        raise ValueError(f"상한은 0 이상이어야 합니다: {upper_bound}")
    if upper_bound == 0:
        return BinaryEncoding(upper_bound=0, weights=())

    num_bits = upper_bound.bit_length()
    weights = [1 << k for k in range(num_bits - 1)]
    last = upper_bound - ((1 << (num_bits - 1)) - 1)
    weights.append(last)

    encoding = BinaryEncoding(upper_bound=upper_bound, weights=tuple(weights))
    if sum(encoding.weights) != upper_bound:
        raise RuntimeError(
            f"binary expansion 계수 합이 상한과 다릅니다: "
            f"{sum(encoding.weights)} != {upper_bound}"
        )
    return encoding
