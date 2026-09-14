# 실험 수행 기록

각 단계에서 발생한 문제와 그 처리, 그리고 QA embedding 관련 상태를 기록한다.

---

## 1. 구현 중 발견하여 수정한 문제

### 1.1 sample-energy 짝맞춤 검증의 스케일 오류

**증상**: SA 결과 교차검증에서 우리가 계산한 QUBO energy와 sampler가 보고한
energy의 차이가 임계값을 넘어 검증이 실패했다.

**원인**: 짝맞춤 오류가 아니라 **부동소수점 상쇄(cancellation)** 였다.
penalty 항의 계수는 `10^9` 규모인데 feasible 해의 energy는 `10^4` 규모까지
상쇄되어 남는다. 따라서 energy 자체를 분모로 쓰면 정상적인 합산 순서 차이도
큰 상대오차로 보인다.

**처리**: 상대오차의 분모를 energy가 아니라 **QUBO 계수의 최대 절댓값**으로
바꾸어 "실제 합산 정밀도" 기준으로 비교하도록 했다. 동시에 검증을 best sample
하나에서 **모든 sample**로 확장하고, energy 최소값의 위치(argmin)가 일치하는지도
함께 확인하도록 강화했다.

현재 최대 상대 mismatch는 `2.7e-11`이고 모든 instance에서 argmin이 일치한다.

> 이 지점은 그냥 임계값을 느슨하게 푸는 것으로 넘어갈 수 있었지만, 그렇게 하면
> 실제 짝맞춤 오류를 놓치게 된다. 스케일을 바로잡고 검증 범위를 넓히는 쪽을
> 택했다.

### 1.2 배치 energy 계산의 메모리 폭증

**증상**: 15×15 MS QUBO(변수 1439개, 항 125,296개)에서 여러 sample의 energy를
한 번에 계산할 때 프로세스가 종료되었다.

**원인**: `(num_samples × num_terms)` 크기의 중간 배열을 만들면
`2000 × 125296 ≈ 2.5억` 원소가 되어 수 GB를 소비한다.

**처리**: `qubo_builder._MAX_CHUNK_ELEMENTS`로 중간 배열 크기 상한을 두고
청크 단위로 계산하도록 했다. 결과값은 동일하며 계산 순서만 달라진다.

### 1.3 실제 instance의 exhaustive validation 불가

**증상**: 명세는 4×4에서 전수 검증을 요구했으나, SS 4×4의 QUBO 변수는 44개,
MS 4×4는 120개여서 `2^44` / `2^120` 열거가 불가능하다.

**처리**: 사전 협의를 거쳐 다음과 같이 확정했다.

- 실제 instance: 전수 열거를 시도하되 한계 초과 시 사유와 함께 SKIP 기록
- 별도의 **2×2 toy instance**(수요 `[3,4]`, 용량 `[6,5]`)에서만 전수 검증 수행
- 대신 실제 instance에는 random validation(2000 샘플 energy 항등식)과
  reference solution encoding 검증을 적용

toy instance는 실험 결과에 포함되지 않으며 변환 정확성 확인에만 쓰인다.

---

## 2. 검증 결과

`results/raw/verification_log.txt`(명세 30절 단계별 검증)와
`results/raw/validation_log.csv`(QUBO 정확성 검증) 참조.

| 검증 항목 | 결과 |
|---|---|
| binary expansion 표현 완전성 | 통과 (실제 등장하는 모든 상한에 대해 전수 확인) |
| energy 항등식 (random) | 통과 (최대 상대오차 `4.4e-14`) |
| reference solution encoding | 통과 (제약 잔차 정확히 0) |
| toy 2×2 exhaustive | 통과 (QUBO ground state = MILP 최적값) |
| `MS ≤ MS-int-q ≤ SS` 대소 관계 | 모든 instance에서 성립 |
| sample-energy 짝맞춤 | 통과 (argmin 전부 일치) |

검증 총 26건 중 26건 통과.

---

## 3. QA 실행 상태

**현재 결과 파일의 QA 상태는 전 instance `NO_QPU_ACCESS`이다.**

이는 실행 환경에 D-Wave API 토큰이 없었기 때문이며, **embedding 실패와는
다른 상태**다. 두 상태를 혼동하지 않도록 구분해서 기록한다.

| 상태 | 의미 |
|---|---|
| `OK` | embedding 성공, QPU 실행 완료 |
| `NOT_EMBEDDABLE` | QPU 그래프에 minor embedding 실패 |
| `NO_QPU_ACCESS` | QPU 접근 불가 (토큰 없음) — embedding 판정 자체를 하지 못함 |
| `QPU_RUN_FAILED` | embedding은 됐으나 QPU 실행 중 오류 |

토큰이 있는 환경에서 `notebooks/05_run_qa.ipynb`를 실행하면 embedding 판정과
QA 결과가 채워지고, `06_compare_results.ipynb`를 다시 실행하면 Figure 1·2·3·5·7에
QA가 반영된다. 다른 코드 수정은 필요 없다.

### embedding에 대한 사전 예상 (검증 필요)

D-Wave Advantage에서 밀집 그래프의 embedding 한계는 대략 170~180 논리변수
수준으로 알려져 있다. 본 실험의 QUBO 크기는 다음과 같다.

| instance | SS 변수 | MS 변수 |
|---|---|---|
| 4×4 | 44 | 120 |
| 6×6 | 79 | 259 |
| 8×8 | 117 | 413 |
| 15×15 | 329 | 1439 |

이 수치만 보면 SS는 6×6까지, MS는 4×4 정도가 경계에 놓인다. 다만 실제 embedding
가능 여부는 그래프의 밀도와 구조에 따라 달라지므로 **여기서 결과를 미리
단정하지 않는다.** 실제 판정은 `05_run_qa.ipynb` 실행 결과로 기록한다.

---

## 4. 관측된 결과 (해석은 06 notebook 참조)

### 4.1 1 unit 이산화 손실이 0

모든 instance에서 `MS-int-q`의 최적값이 연속 `MS`와 정확히 같았다
(이산화 손실 0.000000%). 즉 현재 데이터 설정에서 1 unit precision은 MS의
최적해를 손실 없이 표현한다. 따라서 MS QUBO 해의 gap은 이산화가 아니라
**solver 품질에서 온 것**으로 해석할 수 있다.

이 결론은 연속 Gurobi와 정수-q Gurobi를 **둘 다** reference로 기록했기 때문에
내릴 수 있었다.

### 4.2 feasibility 확보 난이도가 formulation에 따라 크게 다름

SA의 feasible sample 비율:

| instance | SS | MS |
|---|---|---|
| 4×4 | 0.9% | 89.7% |
| 6×6 | 0.2% | 97.7% |
| 8×8 | 0.2% | 90.9% |
| 15×15 | 0.0% | 99.9% |

SS는 15×15에서 feasible 해를 하나도 얻지 못했다. SS의 single assignment 제약
`Σ_j x_ij = 1`은 정확히 하나의 변수만 1이어야 하므로 QUBO 지형에서 만족시키기가
까다로운 반면, MS의 demand 제약 `Σ_j q_ij = d_i`는 여러 비트 조합으로 만족시킬
수 있어 feasible 영역이 훨씬 넓다.

**다만 feasible 비율이 높다고 해 품질이 좋은 것은 아니다.** MS는 feasible 해를
쉽게 찾지만 true gap은 오히려 SS보다 크다(8×8에서 MS 84.3% vs SS 33.4%).
feasibility와 solution quality는 분리해서 보아야 한다.

### 4.3 QUBO 계수의 dynamic range

| instance | 계수 범위 |
|---|---|
| 4×4 | `5.6e+07` |
| 6×6 | `1.2e+08` |
| 8×8 | `7.2e+08` |
| 15×15 | `1.6e+09` |

penalty를 "목적함수 최대 이득"보다 크게 잡는 원칙을 지키면 capacity 제약의
penalty 계수가 `λ d_i d_j` 규모까지 커지기 때문이다. D-Wave의 아날로그 정밀도가
실효 4~5비트 수준임을 감안하면, QA에서 이 dynamic range는 심각한 제약이 된다.

이는 **튜닝 부족이 아니라 penalty 기반 QUBO 변환의 구조적 결과**이며 RQ5의
관찰 대상이다. 결과를 "파라미터를 더 잘 맞췄으면 됐을 문제"로 오해하지 않도록
명시해 둔다.

---

## 5. 하지 않은 것

명세의 금지사항을 지켰음을 명시한다.

- penalty coefficient를 grid search하지 않았다. `margin = 1.1`은 실험 시작 전에
  설정 파일에 고정했다.
- SA/QA 파라미터를 결과를 보고 조정하지 않았다.
- 15×15 QUBO가 크다는 이유로 formulation을 축소하지 않았다.
- QA 결과를 실행한 것처럼 표시하지 않았다. `NO_QPU_ACCESS`를 그대로 기록했다.
- Gurobi 목적값과 QUBO energy를 직접 비교하지 않았다. 모든 solver 해를 원래 CFLP
  변수 공간으로 decode한 뒤 원래 objective를 다시 계산했다.
- 결과를 미리 가정하지 않았다.

### 환경 관련 참고

SA 파라미터(`num_reads=1000`, `num_sweeps=1000`)는 명세대로 유지했다.
`_MAX_CHUNK_ELEMENTS`는 실험 파라미터가 아니라 **동일한 값을 계산하는 방식**에
대한 구현 상수이므로, 이를 조정한 것은 파라미터 튜닝에 해당하지 않는다.
