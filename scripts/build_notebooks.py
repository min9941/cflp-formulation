"""notebooks/ 아래의 6개 실험 notebook을 생성하는 스크립트.

notebook에는 긴 알고리즘 코드를 두지 않고 ``src/`` 모듈을 호출하는 얇은
실행/시각화 코드만 둔다는 원칙을 따른다.

실행:
    python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

BOOTSTRAP = '''\
# 프로젝트 루트를 import 경로에 추가한다.
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path

config = load_config(PROJECT_ROOT / "config" / "experiment_config.yaml")
DATA_DIR = resolve_path(config, "data_dir")
RAW_DIR = resolve_path(config, "raw_dir")
PROCESSED_DIR = resolve_path(config, "processed_dir")
FIGURE_DIR = resolve_path(config, "figure_dir")
SOLUTION_DIR = RAW_DIR / "solutions"
SOLUTION_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)
print("설정 로드 완료:", len(config["instances"]), "개 instance")
'''


def _notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    """(종류, 내용) 목록으로부터 notebook 객체를 만든다."""
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(content)
        if kind == "md"
        else nbf.v4.new_code_cell(content)
        for kind, content in cells
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return notebook


# ---------------------------------------------------------------------------
# 01 — instance 생성
# ---------------------------------------------------------------------------
NB01 = [
    (
        "md",
        "# 01. CFLP instance 생성\n\n"
        "Beasley(OR-Library) / Cornuejols-Sridharan-Thizy(1991) 계열의 CFLP "
        "instance generation 구조를 따라 4개의 instance를 생성한다.\n\n"
        "- 위치: `U[0,1] x U[0,1]`\n"
        "- 수요: `d_i ~ Normal(35, 5)` 를 반올림 후 최소 1로 clip\n"
        "- 용량: `s_j ~ Uniform[10,160]` 을 `sum_j s_j = 1.5 * sum_i d_i` 가 "
        "되도록 정수 rescaling\n"
        "- 고정비용: `f_j = U[0,90] + U[100,110] * sqrt(s_j)` (용량과 양의 상관)\n"
        "- 운송 단가: `c_ij = 10 * dist_ij`\n\n"
        "모든 instance는 설정 파일에 고정된 seed로 생성되므로 재현 가능하다. "
        "**SS와 MS는 동일한 instance data를 사용한다.**",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "from src.data_generator import generate_all_instances\n\n"
        "instances = generate_all_instances(config)\n"
        "for instance in instances:\n"
        "    path = instance.save(DATA_DIR)\n"
        "    print(f\"{instance.name}: 저장 -> {path.name}\")",
    ),
    (
        "md",
        "## 생성된 instance 요약\n\n"
        "전체 용량 / 전체 수요 비율이 목표값 1.5에 정확히 맞는지 확인한다.",
    ),
    (
        "code",
        "summary = pd.DataFrame(\n"
        "    [\n"
        "        {\n"
        "            \"instance\": instance.name,\n"
        "            \"size\": instance.num_customers,\n"
        "            \"seed\": instance.seed,\n"
        "            \"total_demand\": instance.total_demand,\n"
        "            \"total_capacity\": instance.total_capacity,\n"
        "            \"capacity_ratio\": round(instance.capacity_ratio, 4),\n"
        "            \"demand_min\": int(instance.demands.min()),\n"
        "            \"demand_max\": int(instance.demands.max()),\n"
        "            \"capacity_min\": int(instance.capacities.min()),\n"
        "            \"capacity_max\": int(instance.capacities.max()),\n"
        "            \"fixed_cost_mean\": round(float(instance.fixed_costs.mean()), 2),\n"
        "            \"transport_cost_max\": round(float(instance.transport_costs.max()), 4),\n"
        "        }\n"
        "        for instance in instances\n"
        "    ]\n"
        ")\n"
        "summary.to_csv(RAW_DIR / \"instance_summary.csv\", index=False)\n"
        "summary",
    ),
    (
        "md",
        "## 용량-고정비용 상관관계 확인\n\n"
        "고정비용을 모든 시설에 동일하게 주지 않았고, 용량과 양의 상관을 "
        "갖는지 수치로 확인한다.",
    ),
    (
        "code",
        "for instance in instances:\n"
        "    correlation = float(\n"
        "        np.corrcoef(instance.capacities, instance.fixed_costs)[0, 1]\n"
        "    )\n"
        "    print(f\"{instance.name}: corr(s_j, f_j) = {correlation:+.4f}\")",
    ),
    (
        "md",
        "## 데이터 분포 시각화\n\n"
        "위쪽 행은 고객(원, 크기 = 수요)과 시설(사각형, 크기 = 용량)의 위치이고, "
        "아래쪽 행은 용량 대비 고정비용이다.",
    ),
    (
        "code",
        "from src.plotting import plot_instance_overview\n\n"
        "path = plot_instance_overview(instances, FIGURE_DIR)\n"
        "print(\"저장:\", path)\n"
        "from IPython.display import Image\n"
        "Image(filename=str(path))",
    ),
]

# ---------------------------------------------------------------------------
# 02 — Gurobi
# ---------------------------------------------------------------------------
NB02 = [
    (
        "md",
        "# 02. Gurobi로 원래 MILP 해결\n\n"
        "Gurobi 결과는 SA/QA 비교의 **true optimum reference**로 사용한다.\n\n"
        "세 가지 모형을 푼다.\n\n"
        "| 모형 | 변수 | 의미 |\n"
        "|---|---|---|\n"
        "| `SS` | `x_ij in {0,1}` | Single-source |\n"
        "| `MS` | `q_ij` 연속 | 원래 Multiple-source (x_ij in [0,1]과 동일) |\n"
        "| `MS_INT` | `q_ij` 정수 | QUBO와 동일한 1 unit 이산화 수준 |\n\n"
        "`MS`와 `MS_INT`를 모두 기록하면, QUBO 기반 해의 gap에서 "
        "**이산화로 인한 손실**과 **solver 품질로 인한 손실**을 분리할 수 있다.\n\n"
        "linking 제약 `x_ij <= y_j`는 capacity 제약에 의해 함의되는 redundant "
        "제약이므로 SS/MS/QUBO 모두에서 제거하였다(동일한 feasible set).",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "from src import cflp_ms, cflp_ss, gurobi_solver\n"
        "from src.data_generator import CFLPInstance\n"
        "from src.persistence import save_solution, save_table\n\n"
        "instances = [\n"
        "    CFLPInstance.load(DATA_DIR / f\"{spec['name']}.json\")\n"
        "    for spec in config[\"instances\"]\n"
        "]\n"
        "records = []\n"
        "for instance in instances:\n"
        "    references = gurobi_solver.solve_all(instance, config[\"gurobi\"])\n"
        "    for key, result in references.items():\n"
        "        save_solution(result.solution, SOLUTION_DIR, instance.name, f\"gurobi_{key}\")\n"
        "        record = {\"instance\": instance.name, \"size\": instance.num_customers}\n"
        "        record.update(result.to_record())\n"
        "        records.append(record)\n"
        "    print(\n"
        "        f\"{instance.name}: SS={references['SS'].objective:.4f}, \"\n"
        "        f\"MS={references['MS'].objective:.4f}, \"\n"
        "        f\"MS_INT={references['MS_INT'].objective:.4f}\"\n"
        "    )\n\n"
        "gurobi_results = pd.DataFrame(records)\n"
        "save_table(gurobi_results, RAW_DIR, \"gurobi_results.csv\")\n"
        "gurobi_results",
    ),
    (
        "md",
        "## 최적성 확인\n\n"
        "모든 모형이 `OPTIMAL` 상태이고 MIP gap이 0인지 확인한다.",
    ),
    (
        "code",
        "all_optimal = bool((gurobi_results[\"status\"] == \"OPTIMAL\").all())\n"
        "max_gap = float(gurobi_results[\"mip_gap\"].max())\n"
        "print(\"모두 OPTIMAL:\", all_optimal)\n"
        "print(\"최대 MIP gap:\", max_gap)\n"
        "if not all_optimal:\n"
        "    raise RuntimeError(\"OPTIMAL이 아닌 모형이 있습니다. 결과를 reference로 쓸 수 없습니다.\")",
    ),
    (
        "md",
        "## 해의 feasibility 재검증과 대소 관계\n\n"
        "SS는 MS의 해를 이진 배정으로 제한한 문제이므로 다음 관계가 "
        "성립해야 한다.\n\n"
        "$$Obj_{MS} \\le Obj_{MS\\text{-}int} \\le Obj_{SS}$$\n\n"
        "이 관계가 깨지면 formulation 구현에 오류가 있다는 뜻이다.",
    ),
    (
        "code",
        "tolerance = float(config[\"feasibility\"][\"tolerance\"])\n"
        "checks = []\n"
        "for instance in instances:\n"
        "    subset = gurobi_results[gurobi_results[\"instance\"] == instance.name]\n"
        "    values = dict(zip(subset[\"gurobi_model\"], subset[\"objective\"]))\n"
        "    checks.append(\n"
        "        {\n"
        "            \"instance\": instance.name,\n"
        "            \"MS\": round(values[\"MS\"], 4),\n"
        "            \"MS_INT\": round(values[\"MS-int-q\"], 4),\n"
        "            \"SS\": round(values[\"SS\"], 4),\n"
        "            \"ordering_ok\": bool(\n"
        "                values[\"MS\"] <= values[\"MS-int-q\"] + 1e-6\n"
        "                and values[\"MS-int-q\"] <= values[\"SS\"] + 1e-6\n"
        "            ),\n"
        "            \"discretization_loss\": round(values[\"MS-int-q\"] - values[\"MS\"], 6),\n"
        "            \"single_source_premium\": round(values[\"SS\"] - values[\"MS\"], 4),\n"
        "        }\n"
        "    )\n"
        "ordering = pd.DataFrame(checks)\n"
        "save_table(ordering, RAW_DIR, \"gurobi_ordering_check.csv\")\n"
        "ordering",
    ),
]

# ---------------------------------------------------------------------------
# 03 — QUBO 생성 및 검증
# ---------------------------------------------------------------------------
NB03 = [
    (
        "md",
        "# 03. QUBO 생성 및 정확성 검증\n\n"
        "## penalty coefficient 계산\n\n"
        "SS와 MS의 목적함수 계수는 모두 비음수이다 (`f_j >= 0`, `c_ij d_i >= 0`). "
        "따라서 목적함수에서 얻을 수 있는 최대 이득의 보수적 상한은\n\n"
        "$$U_{obj} = \\sum_j f_j + \\sum_i \\sum_j c_{ij} d_i$$\n\n"
        "이다. 정수 계수를 갖는 등식 제약 `g_k(z) = 0`이 위반되면 "
        "`g_k(z)^2 >= 1`이므로 penalty 증가분은 최소 `lambda`이다. 따라서\n\n"
        "$$\\lambda = \\text{margin} \\times U_{obj}, \\qquad \\text{margin} > 1$$\n\n"
        "로 두면 어떤 제약 위반도 목적함수 개선으로 상쇄될 수 없다. "
        "margin은 설정 파일에 고정하며 grid search 하지 않는다.\n\n"
        "MS QUBO에서 `q_ij`의 binary expansion 계수 합이 정확히 `d_i`이므로 "
        "운송비 계수 총합이 SS와 같아진다. 즉 SS와 MS가 **동일한 기준식**으로 "
        "lambda를 계산한다.",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "from src.data_generator import CFLPInstance\n"
        "from src.persistence import load_solution, save_table\n"
        "from src.qubo_builder import build_qubo, qubo_statistics\n\n"
        "margin = float(config[\"penalty\"][\"margin\"])\n"
        "precision = int(config[\"encoding\"][\"precision\"])\n"
        "instances = [\n"
        "    CFLPInstance.load(DATA_DIR / f\"{spec['name']}.json\")\n"
        "    for spec in config[\"instances\"]\n"
        "]\n\n"
        "models = {}\n"
        "rows = []\n"
        "for instance in instances:\n"
        "    for formulation in (\"SS\", \"MS\"):\n"
        "        model = build_qubo(instance, formulation, margin, precision)\n"
        "        models[(instance.name, formulation)] = model\n"
        "        row = {\n"
        "            \"instance\": instance.name,\n"
        "            \"size\": instance.num_customers,\n"
        "            \"formulation\": formulation,\n"
        "        }\n"
        "        row.update(qubo_statistics(model))\n"
        "        rows.append(row)\n\n"
        "qubo_stats = pd.DataFrame(rows)\n"
        "save_table(qubo_stats, RAW_DIR, \"qubo_stats.csv\")\n"
        "qubo_stats",
    ),
    (
        "md",
        "## 변수 수 구성\n\n"
        "- `decision_variables`: 원래 formulation에 이미 binary인 변수\n"
        "- `encoding_variables`: 정수 변수 `q_ij`의 binary expansion 변수\n"
        "- `slack_variables`: capacity 부등식을 등식으로 바꾸기 위한 slack 변수",
    ),
    (
        "code",
        "pivot = qubo_stats.pivot_table(\n"
        "    index=[\"instance\", \"size\"],\n"
        "    columns=\"formulation\",\n"
        "    values=[\"qubo_variables\", \"qubo_terms\"],\n"
        ").sort_index(level=\"size\")\n"
        "pivot[\"MS/SS variable ratio\"] = (\n"
        "    pivot[(\"qubo_variables\", \"MS\")] / pivot[(\"qubo_variables\", \"SS\")]\n"
        ").round(2)\n"
        "pivot",
    ),
    (
        "md",
        "## 검증 1 — binary expansion의 표현 완전성\n\n"
        "마지막 계수를 `U - (2^(K-1) - 1)`로 잘라낸 encoding이 `[0, U]`의 "
        "**모든 정수**를 정확히 표현하는지 전수 확인한다. 이 검증이 없으면 "
        "encoding이 표현할 수 없는 값 때문에 ground state가 구조적으로 "
        "infeasible해질 수 있다.",
    ),
    (
        "code",
        "from src import validation\n\n"
        "bounds = sorted(\n"
        "    {int(value) for instance in instances for value in instance.demands}\n"
        "    | {int(value) for instance in instances for value in instance.capacities}\n"
        "    | set(range(0, 17))\n"
        ")\n"
        "check = validation.validate_encoding_completeness(bounds)\n"
        "print(check)\n"
        "assert check.passed",
    ),
    (
        "md",
        "## 검증 2 — energy 항등식 (random validation)\n\n"
        "무작위 binary 할당 `z`에 대해 다음이 성립해야 한다.\n\n"
        "$$\\text{QUBO energy}(z) = f_{orig}(\\text{decode}(z)) + "
        "\\lambda \\sum_k g_k(z)^2$$\n\n"
        "상수 offset 누락, 계수 합산 오류, 부호 오류가 있으면 즉시 드러난다.\n\n"
        "## 검증 3 — reference solution encoding\n\n"
        "Gurobi 최적해를 QUBO 변수로 encoding했을 때 모든 제약 잔차가 0이고 "
        "QUBO energy가 MILP 목적값과 일치해야 한다.",
    ),
    (
        "code",
        "tolerance = float(config[\"validation\"][\"tolerance\"])\n"
        "num_samples = int(config[\"validation\"][\"random_samples\"])\n"
        "seed = int(config[\"validation\"][\"random_seed\"])\n\n"
        "validation_rows = []\n"
        "for instance in instances:\n"
        "    for formulation in (\"SS\", \"MS\"):\n"
        "        model = models[(instance.name, formulation)]\n"
        "        reference_tag = \"gurobi_SS\" if formulation == \"SS\" else \"gurobi_MS_INT\"\n"
        "        reference_solution = load_solution(SOLUTION_DIR, instance.name, reference_tag)\n"
        "        reference_row = pd.read_csv(RAW_DIR / \"gurobi_results.csv\")\n"
        "        wanted = \"SS\" if formulation == \"SS\" else \"MS-int-q\"\n"
        "        reference_objective = float(\n"
        "            reference_row[\n"
        "                (reference_row[\"instance\"] == instance.name)\n"
        "                & (reference_row[\"gurobi_model\"] == wanted)\n"
        "            ][\"objective\"].iloc[0]\n"
        "        )\n"
        "        for result in (\n"
        "            validation.validate_energy_identity(\n"
        "                model, instance, num_samples, seed, tolerance\n"
        "            ),\n"
        "            validation.validate_reference_solution(\n"
        "                model, instance, reference_solution, reference_objective, tolerance\n"
        "            ),\n"
        "        ):\n"
        "            print(result)\n"
        "            validation_rows.append(\n"
        "                {\n"
        "                    \"instance\": instance.name,\n"
        "                    \"formulation\": formulation,\n"
        "                    \"check\": result.name.split(\"[\")[0],\n"
        "                    \"passed\": result.passed,\n"
        "                    \"message\": result.message,\n"
        "                }\n"
        "            )",
    ),
    (
        "md",
        "## 검증 4 — exhaustive validation\n\n"
        "실제 실험 instance는 QUBO 변수가 44개 이상이라 전수 열거가 불가능하다 "
        "(`2^44` 이상). 따라서 다음 두 가지를 수행한다.\n\n"
        "1. 실제 instance: 전수 열거를 시도하되 한계를 넘으면 사유와 함께 SKIP 기록\n"
        "2. 별도의 2x2 toy instance: 수요/용량을 매우 작게 고정하여 **전수 열거로** "
        "QUBO 최소값이 MILP 최적값과 일치하는지 확인\n\n"
        "toy instance는 실제 실험 결과에 포함되지 않으며 오직 변환 정확성 "
        "검증에만 사용한다.",
    ),
    (
        "code",
        "from src import gurobi_solver\n"
        "from src.data_generator import build_toy_instance\n\n"
        "toy = build_toy_instance(config[\"validation\"][\"toy\"])\n"
        "toy_references = {\n"
        "    \"SS\": gurobi_solver.solve_ss(toy, config[\"gurobi\"]),\n"
        "    \"MS\": gurobi_solver.solve_ms_integer(toy, config[\"gurobi\"]),\n"
        "}\n"
        "for formulation, reference in toy_references.items():\n"
        "    toy_model = build_qubo(toy, formulation, margin, precision)\n"
        "    result = validation.validate_exhaustive(\n"
        "        toy_model, toy, reference.objective, tolerance\n"
        "    )\n"
        "    print(f\"toy {formulation}: QUBO 변수 {toy_model.num_variables}개\")\n"
        "    print(\"   \", result)\n"
        "    validation_rows.append(\n"
        "        {\n"
        "            \"instance\": \"toy2x2\",\n"
        "            \"formulation\": formulation,\n"
        "            \"check\": \"exhaustive\",\n"
        "            \"passed\": result.passed,\n"
        "            \"message\": result.message,\n"
        "        }\n"
        "    )",
    ),
    (
        "code",
        "for instance in instances:\n"
        "    for formulation in (\"SS\", \"MS\"):\n"
        "        model = models[(instance.name, formulation)]\n"
        "        result = validation.validate_exhaustive(model, instance, 1.0, tolerance)\n"
        "        validation_rows.append(\n"
        "            {\n"
        "                \"instance\": instance.name,\n"
        "                \"formulation\": formulation,\n"
        "                \"check\": \"exhaustive\",\n"
        "                \"passed\": result.passed,\n"
        "                \"message\": result.message,\n"
        "            }\n"
        "        )\n"
        "        print(result)",
    ),
    (
        "code",
        "validation_log = pd.DataFrame(validation_rows)\n"
        "save_table(validation_log, RAW_DIR, \"validation_log.csv\")\n"
        "print(\n"
        "    f\"검증 {len(validation_log)}건 중 통과 \"\n"
        "    f\"{int(validation_log['passed'].sum())}건\"\n"
        ")\n"
        "assert bool(validation_log[\"passed\"].all()), \"검증 실패 항목이 있습니다.\"\n"
        "validation_log",
    ),
    (
        "md",
        "## QUBO 계수 통계\n\n"
        "penalty가 목적함수 규모를 압도하도록 설정되므로 계수의 dynamic range가 "
        "매우 커진다. 이는 QA 하드웨어의 아날로그 정밀도와 직결되는 문제이며 "
        "RQ5의 핵심 관찰 대상이다.",
    ),
    (
        "code",
        "qubo_stats[\n"
        "    [\n"
        "        \"instance\",\n"
        "        \"formulation\",\n"
        "        \"penalty_lambda\",\n"
        "        \"qubo_min\",\n"
        "        \"qubo_max\",\n"
        "        \"qubo_range\",\n"
        "    ]\n"
        "].assign(\n"
        "    dynamic_range_orders=lambda frame: np.log10(\n"
        "        frame[\"qubo_range\"].abs().clip(lower=1e-12)\n"
        "    ).round(2)\n"
        ")",
    ),
]

# ---------------------------------------------------------------------------
# 04 — SA
# ---------------------------------------------------------------------------
NB04 = [
    (
        "md",
        "# 04. Simulated Annealing 실행\n\n"
        "`dwave-neal`을 사용하여 QUBO를 직접 해결한다. QA와의 비교가 공정하도록 "
        "**동일한 QUBO**를 사용하며, 파라미터는 설정 파일에 고정하고 결과를 보고 "
        "튜닝하지 않는다.\n\n"
        "best sample 하나만 보지 않고, `num_reads` 안에서 feasible sample 비율과 "
        "feasible 해 중 최선의 목적값도 함께 기록하여 solver의 안정성을 평가한다.\n\n"
        "**주의**: dimod `SampleSet.samples()`는 정렬된 뷰를 반환하므로 "
        "`record.energy`와 인덱스가 어긋날 수 있다. 본 프로젝트는 항상 "
        "`record.sample`과 `record.energy`를 같은 인덱스로 함께 읽는다.",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "from src import sa_solver\n"
        "from src.data_generator import CFLPInstance\n"
        "from src.persistence import save_samples, save_table\n"
        "from src.qubo_builder import build_qubo\n\n"
        "margin = float(config[\"penalty\"][\"margin\"])\n"
        "precision = int(config[\"encoding\"][\"precision\"])\n"
        "feas_tolerance = float(config[\"feasibility\"][\"tolerance\"])\n\n"
        "instances = [\n"
        "    CFLPInstance.load(DATA_DIR / f\"{spec['name']}.json\")\n"
        "    for spec in config[\"instances\"]\n"
        "]\n\n"
        "# linking constraint 포함/제외 두 변형을 모두 실행한다.\n"
        "# x_ij <= y_j 는 capacity constraint에 의해 함의되는 redundant 제약이므로\n"
        "# feasible region은 동일하다. 따라서 관측되는 차이는 전부 QUBO 표현의\n"
        "# 비용이며, 이것이 이 비교의 목적이다.\n"
        "LINKING_VARIANTS = (False, True)\n"
        "\n"
        "records = []\n"
        "best_samples = {}\n"
        "for instance in instances:\n"
        "    for formulation in (\"SS\", \"MS\"):\n"
        "      for include_linking in LINKING_VARIANTS:\n"
        "        model = build_qubo(\n"
        "            instance, formulation, margin, precision,\n"
        "            include_linking=include_linking,\n"
        "        )\n"
        "        outcome = sa_solver.solve(model, instance, config[\"sa\"], feas_tolerance)\n"
        "        record = {\n"
        "            \"instance\": instance.name,\n"
        "            \"size\": instance.num_customers,\n"
        "            \"formulation\": formulation,\n"
        "            \"linking\": include_linking,\n"
        "            \"qubo_variables\": model.num_variables,\n"
        "            \"qubo_quadratic_terms\": model.num_quadratic_terms,\n"
        "        }\n"
        "        record.update(outcome.to_record())\n"
        "        records.append(record)\n"
        "        tag = \"L\" if include_linking else \"N\"\n"
        "        key = f\"{instance.name}__{formulation}__{tag}\"\n"
        "        if outcome.best_sample is not None:\n"
        "            best_samples[key + \"__best\"] = outcome.best_sample\n"
        "        if outcome.best_feasible_sample is not None:\n"
        "            best_samples[key + \"__best_feasible\"] = outcome.best_feasible_sample\n"
        "        print(\n"
        "            f\"{instance.name} {formulation} linking={str(include_linking):5s}: \"\n"
        "            f\"vars={model.num_variables:5d}, \"\n"
        "            f\"obj={outcome.best_objective:.2f}, \"\n"
        "            f\"feasible_fraction={outcome.feasible_fraction:.3f}, \"\n"
        "            f\"time={outcome.runtime:.1f}s\"\n"
        "        )\n\n"
        "sa_results = pd.DataFrame(records)\n"
        "save_table(sa_results, RAW_DIR, \"sa_results.csv\")\n"
        "save_samples(best_samples, RAW_DIR, \"sa_best_samples.npz\")\n"
        "sa_results",
    ),
    (
        "md",
        "## sample-energy 짝맞춤 교차검증\n\n"
        "우리가 직접 계산한 QUBO energy와 sampler가 보고한 energy가 일치하는지 "
        "확인한다. best 하나가 아니라 **모든 sample**에 대해 비교하며, "
        "energy 최소값의 위치(argmin)가 일치하는지도 확인한다.\n\n"
        "penalty 항이 `10^9` 규모에서 `10^4`까지 상쇄되므로, 상대오차의 "
        "분모로는 energy가 아니라 QUBO 계수의 최대 절댓값을 쓴다. 즉 "
        "\"실제 합산 정밀도\" 기준으로 본다.",
    ),
    (
        "code",
        "mismatch = sa_results[\n"
        "    [\"instance\", \"formulation\", \"energy_mismatch\", \"argmin_agreement\"]\n"
        "].copy()\n"
        "print(\"최대 상대 mismatch:\", float(mismatch[\"energy_mismatch\"].max()))\n"
        "print(\"argmin 일치:\", bool(mismatch[\"argmin_agreement\"].all()))\n"
        "assert float(mismatch[\"energy_mismatch\"].max()) < 1e-6\n"
        "assert bool(mismatch[\"argmin_agreement\"].all())\n"
        "mismatch",
    ),
    (
        "md",
        "## 안정성 지표 요약\n\n"
        "`feasible_fraction`은 전체 read 중 원래 CFLP formulation 기준으로 "
        "feasible한 sample의 비율이다.",
    ),
    (
        "code",
        "sa_results[\n"
        "    [\n"
        "        \"instance\",\n"
        "        \"formulation\",\n"
        "        \"linking\",\n"
        "        \"qubo_variables\",\n"
        "        \"feasible_fraction\",\n"
        "        \"best_feasible_objective\",\n"
        "        \"runtime\",\n"
        "    ]\n"
        "]",
    ),
    (
        "md",
        "## linking constraint 비교\n\n"
        "세 가지 비교를 수행한다.\n\n"
        "1. **linking 유무** — 같은 formulation 안에서 포함/제외를 비교한다. "
        "feasible region이 동일하므로 차이는 전부 QUBO 표현의 비용이다.\n"
        "2. **linking 포함끼리** — SS vs MS\n"
        "3. **linking 제외끼리** — SS vs MS (본 실험의 기본 설정)",
    ),
    (
        "code",
        "from src.comparison import linking_effect, formulation_gap\n\n"
        "print(\"[1] linking 유무 (같은 formulation 내)\")\n"
        "display(linking_effect(sa_results))",
    ),
    (
        "code",
        "print(\"[2] linking 포함끼리: SS vs MS\")\n"
        "display(formulation_gap(sa_results, linking=True))\n"
        "print(\"[3] linking 제외끼리: SS vs MS\")\n"
        "display(formulation_gap(sa_results, linking=False))",
    ),
]

# ---------------------------------------------------------------------------
# 05 — QA
# ---------------------------------------------------------------------------
NB05 = [
    (
        "md",
        "# 05. Quantum Annealing 실행\n\n"
        "## 절차\n\n"
        "```\n"
        "QUBO 생성 -> 변수 수 확인 -> QPU embedding 시도\n"
        "   embedding 가능?  YES -> QA 실행\n"
        "                    NO  -> NOT_EMBEDDABLE 기록\n"
        "```\n\n"
        "## 고정 사항 (튜닝하지 않음)\n\n"
        "- chain strength: `alpha * max|J|` 규칙으로 고정. instance마다 QUBO 계수 "
        "스케일이 크게 다르므로, 절대값을 고정하면 오히려 instance 간 비교 조건이 "
        "불공정해진다.\n"
        "- annealing time, num_reads: 설정 파일 고정값\n"
        "- embedding: seed / timeout / tries 고정 → 판정이 재현 가능\n\n"
        "## embedding 실패 시\n\n"
        "QUBO를 축소하거나 변수를 제거하거나 formulation을 바꾸지 않는다. "
        "그대로 `NOT_EMBEDDABLE`로 기록하고, 해당 instance의 Gurobi/SA 결과는 "
        "정상적으로 보고한다.\n\n"
        "**실행 전 준비**: 환경변수 `DWAVE_API_TOKEN`을 설정하거나 "
        "`dwave config create`로 토큰을 등록해야 한다. 토큰이 없으면 이 notebook은 "
        "`NO_QPU_ACCESS`를 기록하고 정상 종료한다.",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "from src import qa_solver\n"
        "from src.data_generator import CFLPInstance\n"
        "from src.persistence import save_samples, save_table\n"
        "from src.qubo_builder import build_qubo\n\n"
        "margin = float(config[\"penalty\"][\"margin\"])\n"
        "precision = int(config[\"encoding\"][\"precision\"])\n"
        "feas_tolerance = float(config[\"feasibility\"][\"tolerance\"])\n\n"
        "instances = [\n"
        "    CFLPInstance.load(DATA_DIR / f\"{spec['name']}.json\")\n"
        "    for spec in config[\"instances\"]\n"
        "]\n\n"
        "LINKING_VARIANTS = (False, True)\n"
        "\n"
        "records = []\n"
        "best_samples = {}\n"
        "for instance in instances:\n"
        "    for formulation in (\"SS\", \"MS\"):\n"
        "      for include_linking in LINKING_VARIANTS:\n"
        "        model = build_qubo(\n"
        "            instance, formulation, margin, precision,\n"
        "            include_linking=include_linking,\n"
        "        )\n"
        "        outcome, embedding = qa_solver.solve(\n"
        "            model,\n"
        "            instance,\n"
        "            config[\"qa\"],\n"
        "            config[\"embedding\"],\n"
        "            feas_tolerance,\n"
        "        )\n"
        "        record = {\n"
        "            \"instance\": instance.name,\n"
        "            \"size\": instance.num_customers,\n"
        "            \"formulation\": formulation,\n"
        "            \"linking\": include_linking,\n"
        "            \"qubo_variables\": model.num_variables,\n"
        "            \"qubo_quadratic_terms\": model.num_quadratic_terms,\n"
        "        }\n"
        "        record.update(outcome.to_record())\n"
        "        records.append(record)\n"
        "        tag = \"L\" if include_linking else \"N\"\n"
        "        key = f\"{instance.name}__{formulation}__{tag}\"\n"
        "        if outcome.best_sample is not None:\n"
        "            best_samples[key + \"__best\"] = outcome.best_sample\n"
        "        if outcome.best_feasible_sample is not None:\n"
        "            best_samples[key + \"__best_feasible\"] = outcome.best_feasible_sample\n"
        "        print(\n"
        "            f\"{instance.name} {formulation} linking={str(include_linking):5s} \"\n"
        "            f\"vars={model.num_variables:5d}: status={outcome.status}\"\n"
        "        )\n"
        "        if outcome.status != \"OK\":\n"
        "            print(\"   \", outcome.extra.get(\"qa_message\", \"\"))\n\n"
        "qa_results = pd.DataFrame(records)\n"
        "save_table(qa_results, RAW_DIR, \"qa_results.csv\")\n"
        "if best_samples:\n"
        "    save_samples(best_samples, RAW_DIR, \"qa_best_samples.npz\")\n"
        "qa_results",
    ),
    (
        "md",
        "## embedding 결과 요약\n\n"
        "instance 크기별 embedding 가능/불가능은 그 자체로 중요한 실험 결과이다.",
    ),
    (
        "code",
        "columns = [\n"
        "    column\n"
        "    for column in (\n"
        "        \"instance\",\n"
        "        \"formulation\",\n"
        "        \"qubo_variables\",\n"
        "        \"status\",\n"
        "        \"embedding_status\",\n"
        "        \"physical_qubits\",\n"
        "        \"max_chain_length\",\n"
        "        \"mean_chain_length\",\n"
        "        \"embedding_search_time\",\n"
        "    )\n"
        "    if column in qa_results.columns\n"
        "]\n"
        "qa_results[columns]",
    ),
    (
        "md",
        "## 실행에 성공한 경우의 지표\n\n"
        "QPU access time과 wall-clock을 구분해 기록한다.",
    ),
    (
        "md",
        "## linking constraint 비교\n\n"
        "QA에서 가장 중요한 관찰 지점은 **embedding 성공 여부가 뒤집히는가**이다. "
        "linking constraint는 feasible region을 바꾸지 않으므로, embedding이 "
        "실패한다면 그것은 순수하게 QUBO 표현이 커진 대가이다.",
    ),
    (
        "code",
        "from src.comparison import embedding_summary, formulation_gap, linking_effect\n\n"
        "print(\"[1] linking 유무 (같은 formulation 내)\")\n"
        "display(linking_effect(qa_results))",
    ),
    (
        "code",
        "print(\"[2] linking 포함끼리: SS vs MS\")\n"
        "display(formulation_gap(qa_results, linking=True))\n"
        "print(\"[3] linking 제외끼리: SS vs MS\")\n"
        "display(formulation_gap(qa_results, linking=False))",
    ),
    (
        "code",
        "print(\"embedding 요약\")\n"
        "embedding_summary(qa_results)",
    ),
    (
        "code",
        "succeeded = qa_results[qa_results[\"status\"] == \"OK\"]\n"
        "if succeeded.empty:\n"
        "    print(\"QA를 실행한 instance가 없습니다. 상태:\", list(qa_results[\"status\"].unique()))\n"
        "else:\n"
        "    columns = [\n"
        "        column\n"
        "        for column in (\n"
        "            \"instance\",\n"
        "            \"formulation\",\n"
        "            \"best_energy\",\n"
        "            \"best_objective\",\n"
        "            \"best_is_feasible\",\n"
        "            \"feasible_fraction\",\n"
        "            \"qa_chain_strength\",\n"
        "            \"qa_chain_break_fraction\",\n"
        "            \"qa_qpu_access_time_us\",\n"
        "            \"qa_wall_clock\",\n"
        "        )\n"
        "        if column in succeeded.columns\n"
        "    ]\n"
        "    display(succeeded[columns])",
    ),
]

# ---------------------------------------------------------------------------
# 06 — 결과 통합
# ---------------------------------------------------------------------------
NB06 = [
    (
        "md",
        "# 06. 결과 통합 및 Figure 생성\n\n"
        "Gurobi / SA / QA 결과를 하나의 표로 합치고 Figure 1~7을 생성한다.\n\n"
        "## 비교 원칙\n\n"
        "- 모든 solver 결과는 **원래 CFLP objective space**로 decode하여 비교한다. "
        "QUBO energy만 비교하지 않는다.\n"
        "- true optimality gap의 기준은 각 formulation의 Gurobi 최적값이다.\n\n"
        "$$Gap_{true} = \\frac{Obj_{solver} - Obj_{Gurobi}}{|Obj_{Gurobi}|} \\times 100$$\n\n"
        "- SA/QA의 목적값으로는 **feasible sample 중 최선**을 사용한다. "
        "infeasible한 해의 목적값은 제약을 지키지 않아 얻은 것이므로 "
        "solution quality로 볼 수 없다.\n"
        "- QUBO energy gap도 별도로 기록하되, 핵심 성능 비교에는 "
        "true-optimum gap을 우선한다.",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "from src.persistence import load_table, save_table\n\n"
        "gurobi_results = load_table(RAW_DIR, \"gurobi_results.csv\")\n"
        "qubo_stats = load_table(RAW_DIR, \"qubo_stats.csv\")\n"
        "sa_results = load_table(RAW_DIR, \"sa_results.csv\")\n"
        "try:\n"
        "    qa_results = load_table(RAW_DIR, \"qa_results.csv\")\n"
        "except FileNotFoundError:\n"
        "    qa_results = pd.DataFrame()\n"
        "    print(\"QA 결과 파일이 없습니다. QA 없이 통합합니다.\")\n"
        "print(\n"
        "    \"불러온 행 수:\",\n"
        "    len(gurobi_results), len(qubo_stats), len(sa_results), len(qa_results),\n"
        ")",
    ),
    (
        "md",
        "## 참조 최적값 정리\n\n"
        "- SS의 기준: Gurobi `SS`\n"
        "- MS의 기준: Gurobi `MS` (원래 연속 formulation)\n"
        "- 추가로 `MS-int-q`를 기록하여 1 unit 이산화 손실을 분리한다.",
    ),
    (
        "code",
        "reference = gurobi_results.pivot_table(\n"
        "    index=[\"instance\", \"size\"], columns=\"gurobi_model\", values=\"objective\"\n"
        ").reset_index()\n"
        "reference = reference.rename(\n"
        "    columns={\"SS\": \"ref_SS\", \"MS\": \"ref_MS\", \"MS-int-q\": \"ref_MS_INT\"}\n"
        ")\n"
        "reference[\"discretization_loss_percent\"] = (\n"
        "    (reference[\"ref_MS_INT\"] - reference[\"ref_MS\"])\n"
        "    / reference[\"ref_MS\"].abs()\n"
        "    * 100\n"
        ").round(6)\n"
        "reference",
    ),
    (
        "code",
        "def reference_objective(row: pd.Series) -> float:\n"
        "    \"\"\"formulation에 맞는 Gurobi true optimum을 반환한다.\"\"\"\n"
        "    matched = reference[reference[\"instance\"] == row[\"instance\"]].iloc[0]\n"
        "    return float(matched[\"ref_SS\" if row[\"formulation\"] == \"SS\" else \"ref_MS\"])\n\n\n"
        "def build_solver_rows(frame: pd.DataFrame, solver: str) -> pd.DataFrame:\n"
        "    \"\"\"SA/QA 결과를 공통 스키마로 변환한다.\"\"\"\n"
        "    if frame.empty:\n"
        "        return pd.DataFrame()\n"
        "    rows = frame.copy()\n"
        "    rows[\"solver\"] = solver\n"
        "    rows[\"gurobi_optimum\"] = rows.apply(reference_objective, axis=1)\n"
        "    # feasible 해가 없으면 목적값을 NaN으로 둔다 (성능으로 인정하지 않는다).\n"
        "    rows[\"objective\"] = rows[\"best_feasible_objective\"]\n"
        "    rows[\"is_feasible\"] = rows[\"best_feasible_objective\"].notna()\n"
        "    rows[\"total_violation\"] = rows[\"best_total_violation\"]\n"
        "    return rows\n\n\n"
        "solver_rows = pd.concat(\n"
        "    [build_solver_rows(sa_results, \"SA\"), build_solver_rows(qa_results, \"QA\")],\n"
        "    ignore_index=True,\n"
        ")\n"
        "solver_rows[\"true_gap_percent\"] = (\n"
        "    (solver_rows[\"objective\"] - solver_rows[\"gurobi_optimum\"])\n"
        "    / solver_rows[\"gurobi_optimum\"].abs()\n"
        "    * 100\n"
        ")\n"
        "print(len(solver_rows), \"개 solver 결과\")",
    ),
    (
        "code",
        "gurobi_rows = []\n"
        "for formulation in (\"SS\", \"MS\"):\n"
        "    model_name = \"SS\" if formulation == \"SS\" else \"MS\"\n"
        "    subset = gurobi_results[gurobi_results[\"gurobi_model\"] == model_name].copy()\n"
        "    subset[\"formulation\"] = formulation\n"
        "    subset[\"solver\"] = \"Gurobi\"\n"
        "    subset[\"gurobi_optimum\"] = subset[\"objective\"]\n"
        "    subset[\"true_gap_percent\"] = 0.0\n"
        "    subset[\"is_feasible\"] = True\n"
        "    subset[\"total_violation\"] = 0.0\n"
        "    subset[\"feasible_fraction\"] = 1.0\n"
        "    gurobi_rows.append(subset)\n"
        "gurobi_rows = pd.concat(gurobi_rows, ignore_index=True)\n\n"
        "all_results = pd.concat([gurobi_rows, solver_rows], ignore_index=True)\n"
        "all_results = all_results.merge(\n"
        "    qubo_stats.drop(columns=[\"size\"]), on=[\"instance\", \"formulation\"], how=\"left\"\n"
        ")\n"
        "if \"embedding_status\" not in all_results.columns:\n"
        "    all_results[\"embedding_status\"] = np.nan\n"
        "all_results[\"embedding_status\"] = all_results[\"embedding_status\"].fillna(\n"
        "    all_results[\"solver\"].map({\"Gurobi\": \"N/A\", \"SA\": \"N/A\"})\n"
        ")\n"
        "save_table(all_results, PROCESSED_DIR, \"all_results.csv\")\n"
        "print(\"저장:\", PROCESSED_DIR / \"all_results.csv\")\n"
        "all_results[[\"instance\", \"formulation\", \"solver\", \"status\", \"objective\", \"true_gap_percent\"]]",
    ),
    (
        "md",
        "## 최종 비교표\n\n"
        "요청된 형태의 `formulation x instance x solver` 비교표이다.",
    ),
    (
        "code",
        "comparison = all_results.pivot_table(\n"
        "    index=[\"formulation\", \"size\", \"instance\"],\n"
        "    columns=\"solver\",\n"
        "    values=[\"objective\", \"true_gap_percent\", \"runtime\"],\n"
        "    dropna=False,\n"
        ").sort_index(level=[\"formulation\", \"size\"])\n"
        "save_table(comparison.reset_index(), PROCESSED_DIR, \"comparison_table.csv\")\n"
        "comparison.round(3)",
    ),
    (
        "md",
        "## QA embedding 표\n\n"
        "Figure 7의 기초 자료이다.",
    ),
    (
        "code",
        "if qa_results.empty:\n"
        "    embedding_table = qubo_stats[[\"instance\", \"size\", \"formulation\"]].copy()\n"
        "    embedding_table[\"embedding_status\"] = \"NOT_ATTEMPTED\"\n"
        "else:\n"
        "    embedding_table = qa_results[[\"instance\", \"size\", \"formulation\", \"status\"]].copy()\n"
        "    embedding_table = embedding_table.rename(columns={\"status\": \"embedding_status\"})\n"
        "save_table(embedding_table, PROCESSED_DIR, \"embedding_table.csv\")\n"
        "embedding_table",
    ),
    (
        "md",
        "## Figure 1 ~ 7 생성",
    ),
    (
        "code",
        "from src.plotting import generate_all_figures\n\n"
        "paths = generate_all_figures(all_results, qubo_stats, embedding_table, FIGURE_DIR)\n"
        "for path in paths:\n"
        "    print(\"저장:\", path.name)",
    ),
    (
        "code",
        "from IPython.display import Image, display\n\n"
        "for path in paths:\n"
        "    display(Image(filename=str(path)))",
    ),
    (
        "md",
        "## 결과 요약\n\n"
        "아래 요약은 관측된 수치만 기술한다. runtime이 짧다는 이유만으로 "
        "우열을 단정하지 않으며, solution quality / exactness / feasibility / "
        "classical runtime / QPU runtime / scalability / QUBO size / embedding "
        "feasibility를 구분해서 본다. 또한 이 실험은 QA parameter tuning 실험이 "
        "아니므로 특정 QA 파라미터가 최적이라는 결론을 내리지 않는다.",
    ),
    (
        "code",
        "lines = []\n"
        "lines.append(\"[QUBO 크기]\")\n"
        "for _, row in qubo_stats.sort_values([\"size\", \"formulation\"]).iterrows():\n"
        "    lines.append(\n"
        "        f\"  {row['instance']:>6s} {row['formulation']}: \"\n"
        "        f\"변수 {int(row['qubo_variables']):5d}, 항 {int(row['qubo_terms']):7d}, \"\n"
        "        f\"계수 범위 {row['qubo_range']:.3e}\"\n"
        "    )\n"
        "lines.append(\"\")\n"
        "lines.append(\"[이산화 손실 (MS-int-q vs MS)]\")\n"
        "for _, row in reference.iterrows():\n"
        "    lines.append(\n"
        "        f\"  {row['instance']:>6s}: {row['discretization_loss_percent']:.6f}%\"\n"
        "    )\n"
        "lines.append(\"\")\n"
        "lines.append(\"[SA / QA true optimality gap]\")\n"
        "for _, row in solver_rows.sort_values([\"solver\", \"formulation\", \"size\"]).iterrows():\n"
        "    gap = row[\"true_gap_percent\"]\n"
        "    status = str(row[\"status\"])\n"
        "    if status != \"OK\":\n"
        "        # 실행 자체를 못 한 경우와 '실행했지만 feasible 해가 없음'을\n"
        "        # 반드시 구분해서 적는다.\n"
        "        detail = f\"미실행 ({status})\"\n"
        "    elif pd.isna(gap):\n"
        "        detail = \"실행했으나 feasible 해 없음\"\n"
        "    else:\n"
        "        detail = (\n"
        "            f\"gap {gap:8.3f}%, feasible 비율 \"\n"
        "            f\"{row.get('feasible_fraction', float('nan')):.3f}\"\n"
        "        )\n"
        "    lines.append(\n"
        "        f\"  {row['solver']:>2s} {row['formulation']} {row['instance']:>6s}: {detail}\"\n"
        "    )\n"
        "summary_text = \"\\n\".join(lines)\n"
        "(PROCESSED_DIR / \"summary.txt\").write_text(summary_text, encoding=\"utf-8\")\n"
        "print(summary_text)",
    ),
]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    specifications = {
        "01_generate_instances.ipynb": NB01,
        "02_solve_with_gurobi.ipynb": NB02,
        "03_build_and_validate_qubo.ipynb": NB03,
        "04_run_sa.ipynb": NB04,
        "05_run_qa.ipynb": NB05,
        "06_compare_results.ipynb": NB06,
    }
    for filename, cells in specifications.items():
        path = NOTEBOOK_DIR / filename
        nbf.write(_notebook(cells), path)
        print(f"생성: {path.relative_to(PROJECT_ROOT)} ({len(cells)} 셀)")


if __name__ == "__main__":
    main()
