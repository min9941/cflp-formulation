"""명세 30절의 단계별 검증 절차를 순서대로 실행하는 스크립트.

각 단계에서 오류가 발견되면 다음 instance 크기로 넘어가지 않는다.

실행:
    python scripts/verify_pipeline.py            # 모든 크기
    python scripts/verify_pipeline.py --only 4x4 # 특정 크기만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import cflp_ms, cflp_ss, gurobi_solver, sa_solver, validation
from src.config import load_config
from src.data_generator import build_toy_instance, generate_instance
from src.decoder import decode
from src.qubo_builder import build_qubo, qubo_statistics


def _print_step(number: int, title: str) -> None:
    print(f"\n--- Step {number}: {title} ---")


def run_toy_exhaustive(config: dict) -> bool:
    """toy 2x2 instance에 대한 전수 검증 (Step 7)."""
    toy = build_toy_instance(config["validation"]["toy"])
    gurobi_config = config["gurobi"]
    tolerance = float(config["validation"]["tolerance"])
    margin = float(config["penalty"]["margin"])

    ss_reference = gurobi_solver.solve_ss(toy, gurobi_config)
    ms_reference = gurobi_solver.solve_ms_integer(toy, gurobi_config)

    checks: list[validation.ValidationResult] = []
    for formulation, reference in (("SS", ss_reference), ("MS", ms_reference)):
        model = build_qubo(toy, formulation, margin)
        checks.append(
            validation.validate_exhaustive(
                model, toy, reference.objective, tolerance
            )
        )
    for check in checks:
        print("   ", check)
    return all(check.passed for check in checks)


def verify_instance(name: str, size: int, seed: int, config: dict) -> bool:
    """하나의 instance에 대해 Step 1 ~ 11을 수행한다."""
    print(f"\n{'=' * 70}\n인스턴스 {name} 검증\n{'=' * 70}")
    tolerance = float(config["validation"]["tolerance"])
    feas_tolerance = float(config["feasibility"]["tolerance"])
    margin = float(config["penalty"]["margin"])

    _print_step(1, "instance 생성")
    instance = generate_instance(name, size, seed, config["data"])
    print(
        f"    |I|={instance.num_customers}, |J|={instance.num_facilities}, "
        f"sum d={instance.total_demand}, sum s={instance.total_capacity}, "
        f"ratio={instance.capacity_ratio:.4f}"
    )

    _print_step(2, "Gurobi로 SS / MS / MS-int-q 해결")
    references = gurobi_solver.solve_all(instance, config["gurobi"])
    for key, result in references.items():
        print(
            f"    {key:7s} obj={result.objective:12.4f}  status={result.status}  "
            f"time={result.runtime:.3f}s  gap={result.mip_gap:.2e}"
        )

    _print_step(3, "Gurobi 해의 feasibility 및 대소 관계 확인")
    ss_report = cflp_ss.check_feasibility(
        instance, references["SS"].solution, feas_tolerance
    )
    ms_report = cflp_ms.check_feasibility(
        instance, references["MS"].solution, feas_tolerance
    )
    ms_int_report = cflp_ms.check_feasibility(
        instance, references["MS_INT"].solution, feas_tolerance, require_integral=True
    )
    print(f"    SS feasible={ss_report.is_feasible}")
    print(f"    MS feasible={ms_report.is_feasible}")
    print(f"    MS-int feasible={ms_int_report.is_feasible}")

    ordering_ok = (
        references["MS"].objective <= references["MS_INT"].objective + 1e-6
        and references["MS_INT"].objective <= references["SS"].objective + 1e-6
    )
    print(
        f"    기대 관계 MS <= MS-int <= SS : {ordering_ok} "
        f"({references['MS'].objective:.4f} <= "
        f"{references['MS_INT'].objective:.4f} <= "
        f"{references['SS'].objective:.4f})"
    )
    if not (
        ss_report.is_feasible
        and ms_report.is_feasible
        and ms_int_report.is_feasible
        and ordering_ok
    ):
        print("    [FAIL] Gurobi 단계에서 문제가 발견되었습니다.")
        return False

    models = {}
    for step, formulation in ((4, "SS"), (5, "MS")):
        _print_step(step, f"{formulation} QUBO 생성")
        model = build_qubo(
            instance, formulation, margin, int(config["encoding"]["precision"])
        )
        models[formulation] = model
        stats = qubo_statistics(model)
        print(
            f"    변수={stats['qubo_variables']} "
            f"(decision={stats['decision_variables']}, "
            f"encoding={stats['encoding_variables']}, "
            f"slack={stats['slack_variables']}), "
            f"항={stats['qubo_terms']}, lambda={stats['penalty_lambda']:.4e}"
        )
        print(
            f"    계수 범위: [{stats['qubo_min']:.4e}, {stats['qubo_max']:.4e}]"
        )

    _print_step(6, "QUBO variable mapping 및 energy 항등식 확인")
    all_ok = True
    for formulation, model in models.items():
        reference = references["SS" if formulation == "SS" else "MS_INT"]
        checks = [
            validation.validate_energy_identity(
                model,
                instance,
                int(config["validation"]["random_samples"]),
                int(config["validation"]["random_seed"]),
                tolerance,
            ),
            validation.validate_reference_solution(
                model, instance, reference.solution, reference.objective, tolerance
            ),
        ]
        for check in checks:
            print("   ", check)
            all_ok = all_ok and check.passed
    if not all_ok:
        print("    [FAIL] QUBO 변환 검증에 실패했습니다.")
        return False

    _print_step(7, "toy instance 전수 검증 + 실제 instance 전수 검증 가능 여부")
    if not run_toy_exhaustive(config):
        print("    [FAIL] toy 전수 검증에 실패했습니다.")
        return False
    for formulation, model in models.items():
        skip_check = validation.validate_exhaustive(
            model, instance, references["SS" if formulation == "SS" else "MS_INT"].objective, tolerance
        )
        print("   ", skip_check)

    _print_step(8, "SA로 QUBO 해결")
    outcomes = {}
    for formulation, model in models.items():
        outcome = sa_solver.solve(model, instance, config["sa"], feas_tolerance)
        outcomes[formulation] = outcome
        print(
            f"    {formulation}: energy={outcome.best_energy:.4f}, "
            f"obj={outcome.best_objective:.4f}, "
            f"feasible={outcome.best_is_feasible}, "
            f"feasible_fraction={outcome.feasible_fraction:.3f}, "
            f"time={outcome.runtime:.2f}s"
        )
        # 상대 기준. penalty 항이 1e7 규모에서 상쇄되므로 순수 부동소수점
        # 오차만으로도 1e-9 수준이 나온다. 짝맞춤이 실제로 어긋나면 상대
        # 오차가 O(1)이 되므로 1e-6이면 충분히 민감한 임계값이다.
        if outcome.extra["energy_mismatch"] > 1e-6:
            print(
                f"    [FAIL] sample-energy 짝맞춤 오류 "
                f"(상대 mismatch={outcome.extra['energy_mismatch']:.3e})"
            )
            return False

    _print_step(9, "SA 해 decode")
    for formulation, model in models.items():
        outcome = outcomes[formulation]
        solution = decode(model, outcome.best_sample)
        if formulation == "SS":
            opened = int(np.rint(solution.y).sum())
            print(f"    SS 개설 시설 수 = {opened}")
        else:
            summary = cflp_ms.summarize_solution(instance, solution)
            print(f"    MS 요약 = {summary}")

    _print_step(10, "원래 CFLP feasibility 재검증 및 true gap 계산")
    from src.evaluation import true_optimality_gap

    for formulation in ("SS", "MS"):
        outcome = outcomes[formulation]
        reference = references[formulation]
        gap = true_optimality_gap(outcome.best_feasible_objective, reference.objective)
        print(
            f"    {formulation}: best feasible obj="
            f"{outcome.best_feasible_objective}, true gap={gap:.4f}%"
        )

    _print_step(11, "QA embedding 시도")
    print("    (QPU 토큰이 있는 환경에서 notebooks/05_run_qa.ipynb로 수행)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="단계별 파이프라인 검증")
    parser.add_argument("--only", type=str, default=None, help="검증할 instance 이름")
    arguments = parser.parse_args()

    config = load_config()

    print("=" * 70)
    print("사전 검증: binary expansion 표현 완전성")
    print("=" * 70)
    bounds = list(range(0, 65)) + [100, 127, 128, 160, 255]
    check = validation.validate_encoding_completeness(bounds)
    print("   ", check)
    if not check.passed:
        return 1

    for spec in config["instances"]:
        if arguments.only and spec["name"] != arguments.only:
            continue
        ok = verify_instance(
            spec["name"], int(spec["size"]), int(spec["seed"]), config
        )
        if not ok:
            print(f"\n[중단] {spec['name']}에서 오류가 발견되어 다음 크기로 "
                  f"넘어가지 않습니다.")
            return 1
        print(f"\n[OK] {spec['name']} 검증 통과")

    print("\n모든 검증을 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
