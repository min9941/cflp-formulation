"""notebooks/12, 13 (constraint별 penalty 균형 실험)을 생성한다.

    12_lambda_balance_experiment.ipynb : ratio grid 실행 (오프라인/QPU)
    13_lambda_balance_figures.ipynb    : Figure 12~15 생성

실행:
    python scripts/build_balance_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

BOOTSTRAP = '''\
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

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
print("설정 로드 완료")
'''


def _notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c)
        for k, c in cells
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return notebook


NB12 = [
    (
        "md",
        "# 12. Constraint별 penalty coefficient 균형 실험\n\n"
        "## 문제 인식\n\n"
        "본 실험(notebook 04, 05)은 모든 constraint에 **동일한 lambda**를 "
        "사용했습니다. 그런데 constraint마다 계수의 크기가 다르므로, 같은 "
        "lambda를 써도 하드웨어가 보는 제약 강도는 크게 달라집니다.\n\n"
        "| constraint | 식 | 제곱 후 계수 규모 |\n"
        "|---|---|---|\n"
        "| capacity | `sum_i d_i x_ij - s_j y_j + slack = 0` | `lambda * s_j^2` |\n"
        "| assignment | `sum_j x_ij - 1 = 0` | `lambda` |\n\n"
        "두 penalty의 비는 대략 `2 / s_max^2`입니다. 8x8에서는 약 `10^-4`, "
        "즉 **assignment가 capacity보다 1만 배 약합니다.**\n\n"
        "auto_scale 이후로 보면 assignment 계수가 `4.3e-4`인데 ICE 노이즈는 "
        "`10^-2` 수준입니다. **QPU는 capacity만 보고 assignment는 사실상 보지 "
        "못합니다.** SS의 feasible 비율이 0.2%였던 이유가 여기 있을 수 있습니다.\n\n"
        "## 이 실험이 확인하는 것\n\n"
        "capacity penalty를 고정한 채 assignment penalty에만 배수를 곱해가며 "
        "무엇이 달라지는지 봅니다.\n\n"
        "## 원칙 위반이 아닌 이유\n\n"
        "penalty method가 요구하는 것은 각 constraint에 대해 `lambda_k > Z_ub`"
        "입니다. **모든 constraint가 같은 lambda를 써야 한다는 요구는 "
        "없습니다.** assignment penalty를 키우는 것은 그 조건을 더 강하게 만들 "
        "뿐이므로 ground state가 feasible이라는 보장은 유지됩니다.",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        "## 실험 설정\n\n"
        "아래 네 값만 바꾸면 다른 조건으로 다시 돌릴 수 있습니다.\n\n"
        "- `QA_DRYRUN` — **True면 오프라인, False면 실제 QPU**\n"
        "  - 오프라인은 Pegasus P16 그래프에 minorminer를 직접 실행해 "
        "embedding을 확인하고, sampling은 SA로 대체합니다. "
        "**노이즈를 주지 않으므로 이상적 조건의 상한**을 봅니다.\n"
        "  - 두 모드의 차이가 곧 하드웨어가 잃는 양입니다.\n"
        "- `CAPACITY_MULTIPLIER` — capacity penalty에 쓸 배수. `U_obj`에 "
        "곱해지며 기존 `config['penalty']['margin']`과 같은 의미입니다. "
        "**1.1이면 기존 실험이 그대로 재현됩니다.**\n"
        "- `RATIOS` — assignment penalty가 capacity penalty의 몇 배인지. "
        "`1.0`이 기존 설정입니다.\n"
        "- `TARGETS` — 실험할 (instance, formulation) 목록",
    ),
    (
        "code",
        "QA_DRYRUN = True          # True: 오프라인(노이즈 없음) / False: 실제 QPU\n"
        "CAPACITY_MULTIPLIER = 1.1  # U_obj에 곱하는 배수. 1.1 = 기존 실험 설정\n"
        "RATIOS = (1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)\n"
        "TARGETS = [(\"4x4\", \"SS\"), (\"8x8\", \"SS\")]\n"
        "\n"
        "print(f\"모드              : {'오프라인 (Pegasus + SA, 노이즈 없음)' if QA_DRYRUN else '실제 QPU'}\")\n"
        "print(f\"capacity 배수     : {CAPACITY_MULTIPLIER}\")\n"
        "print(f\"assignment 비율   : {RATIOS}\")\n"
        "print(f\"대상              : {TARGETS}\")",
    ),
    (
        "md",
        "## 공짜로 올릴 수 있는 상한\n\n"
        "assignment penalty를 키워도 **일정 지점까지는 coefficient range가 "
        "전혀 변하지 않습니다.** capacity의 최대 계수가 `lambda * s_max^2`이고 "
        "assignment의 최대 계수가 `2 * lambda * ratio`이므로, "
        "`ratio < s_max^2 / 2`인 동안은 여전히 capacity가 최대이기 "
        "때문입니다.\n\n"
        "그 지점을 넘으면 assignment가 최대가 되어 range가 커지기 시작합니다. "
        "즉 아래 값이 **비용 없이 올릴 수 있는 상한**입니다.",
    ),
    (
        "code",
        "from src.data_generator import CFLPInstance\n"
        "from src.cflp_ss import objective_upper_bound\n"
        "from src import penalty_balance as PB\n\n"
        "gurobi = pd.read_csv(RAW_DIR / \"gurobi_results.csv\")\n"
        "rows = []\n"
        "for name, formulation in TARGETS:\n"
        "    instance = CFLPInstance.load(DATA_DIR / f\"{name}.json\")\n"
        "    model_name = \"SS\" if formulation == \"SS\" else \"MS\"\n"
        "    reference = float(\n"
        "        gurobi[(gurobi[\"instance\"] == name)\n"
        "               & (gurobi[\"gurobi_model\"] == model_name)][\"objective\"].iloc[0]\n"
        "    )\n"
        "    upper = objective_upper_bound(instance)\n"
        "    rows.append({\n"
        "        \"instance\": name,\n"
        "        \"formulation\": formulation,\n"
        "        \"Z_ub\": round(reference, 1),\n"
        "        \"U_obj\": round(upper, 1),\n"
        "        \"lambda_capacity\": round(CAPACITY_MULTIPLIER * upper, 1),\n"
        "        \"lambda/Z_ub\": round(CAPACITY_MULTIPLIER * upper / reference, 2),\n"
        "        \"free_ratio_limit\": round(PB.balanced_ratio(instance), 0),\n"
        "    })\n"
        "pd.DataFrame(rows)",
    ),
    (
        "md",
        "## 실행\n\n"
        "embedding은 grid의 모든 점에서 동일하므로 **한 번만 계산해 "
        "재사용**합니다. penalty 값은 계수의 크기만 바꿀 뿐 어떤 변수쌍이 "
        "연결되는지는 바꾸지 않기 때문입니다. 이렇게 해야 관측된 차이가 "
        "embedding 운이 아니라 penalty 균형 때문임을 보장할 수 있습니다.\n\n"
        "오프라인 모드에서 Pegasus embedding 탐색은 큰 instance에서 수 분이 "
        "걸릴 수 있습니다. 빠르게 확인만 하려면 `check_embedding=False`로 "
        "두십시오.",
    ),
    (
        "code",
        "from src.persistence import save_table\n\n"
        "records = []\n"
        "for name, formulation in TARGETS:\n"
        "    instance = CFLPInstance.load(DATA_DIR / f\"{name}.json\")\n"
        "    model_name = \"SS\" if formulation == \"SS\" else \"MS\"\n"
        "    reference = float(\n"
        "        gurobi[(gurobi[\"instance\"] == name)\n"
        "               & (gurobi[\"gurobi_model\"] == model_name)][\"objective\"].iloc[0]\n"
        "    )\n"
        "    print(f\"--- {name} {formulation} ---\")\n"
        "    points = PB.run_ratio_sweep(\n"
        "        instance=instance,\n"
        "        formulation=formulation,\n"
        "        reference_objective=reference,\n"
        "        config=config,\n"
        "        capacity_multiplier=CAPACITY_MULTIPLIER,\n"
        "        ratios=RATIOS,\n"
        "        qa_dryrun=QA_DRYRUN,\n"
        "        check_embedding=True,\n"
        "    )\n"
        "    for point in points:\n"
        "        print(\n"
        "            f\"  ratio={point.ratio:8.0f}  \"\n"
        "            f\"feasible={point.feasible_fraction * 100:5.1f}%  \"\n"
        "            f\"gap={point.true_gap_percent:7.1f}%  \"\n"
        "            f\"range={point.qubo_range:.3e}\"\n"
        "        )\n"
        "    records.extend(points)\n\n"
        "results = pd.DataFrame([point.to_record() for point in records])",
    ),
    (
        "md",
        "## 결과 저장\n\n"
        "오프라인과 QPU 결과를 **같은 파일에 누적**합니다. `QA_DRYRUN`을 바꿔 "
        "다시 실행하면 두 모드가 함께 쌓여 notebook 13에서 나란히 비교할 수 "
        "있습니다.",
    ),
    (
        "code",
        "OUTPUT = PROCESSED_DIR / \"penalty_balance.csv\"\n"
        "if OUTPUT.exists():\n"
        "    previous = pd.read_csv(OUTPUT)\n"
        "    key = [\"instance\", \"formulation\", \"mode\", \"capacity_multiplier\", \"ratio\"]\n"
        "    merged = pd.concat([previous, results], ignore_index=True)\n"
        "    merged = merged.drop_duplicates(subset=key, keep=\"last\")\n"
        "else:\n"
        "    merged = results\n"
        "save_table(merged, PROCESSED_DIR, \"penalty_balance.csv\")\n"
        "print(f\"저장: {OUTPUT} ({len(merged)} 행)\")\n"
        "print(merged.groupby([\"mode\", \"instance\", \"formulation\"]).size())",
    ),
    (
        "md",
        "## 요약\n\n"
        "`ratio = 1`(기존 설정) 대비 가장 좋았던 ratio를 비교합니다. "
        "`range`가 함께 표시되므로, 개선이 coefficient range를 늘리지 않고 "
        "얻어진 것인지 바로 확인할 수 있습니다.",
    ),
    (
        "code",
        "for line in PB.summarize(merged):\n"
        "    print(line)",
    ),
    (
        "md",
        "## embedding 정보\n\n"
        "오프라인 모드에서는 Pegasus P16(결함 없는 이상적 그래프)에 대한 "
        "결과입니다. 실제 QPU는 결함 큐빗이 있어 물리 큐빗 수가 다소 다를 수 "
        "있습니다.",
    ),
    (
        "code",
        "columns = [\n"
        "    column\n"
        "    for column in (\n"
        "        \"instance\", \"formulation\", \"mode\", \"qubo_variables\",\n"
        "        \"qubo_quadratic_terms\", \"embedding_status\", \"physical_qubits\",\n"
        "        \"max_chain_length\", \"embedding_search_time\",\n"
        "    )\n"
        "    if column in merged.columns\n"
        "]\n"
        "merged[columns].drop_duplicates(subset=[\"instance\", \"formulation\", \"mode\"])",
    ),
]

NB13 = [
    (
        "md",
        "# 13. Penalty 균형 실험 결과 시각화\n\n"
        "notebook 12의 결과로 Figure 12~15를 만듭니다.\n\n"
        "| Figure | 내용 | 확인할 것 |\n"
        "|---|---|---|\n"
        "| 12 | 비율 대 feasible 비율 | 개선이 있는가, 최적 비율은 어디인가 |\n"
        "| 13 | 비율 대 optimality gap | feasibility 개선이 품질로 이어지는가 |\n"
        "| 14 | 정규화된 제약 강도와 ICE 노이즈 | **불균형을 직접 보여주는 그림** |\n"
        "| 15 | 비율 대 coefficient range | 어디까지가 공짜인가 |\n\n"
        "오프라인과 QPU 결과가 모두 있으면 Figure 12, 13에 함께 표시됩니다. "
        "**두 곡선의 차이가 곧 하드웨어가 잃는 양**입니다.",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "from src.persistence import load_table\n"
        "from src import balance_plotting as BP\n"
        "from src import penalty_balance as PB\n\n"
        "results = load_table(PROCESSED_DIR, \"penalty_balance.csv\")\n"
        "print(results.groupby([\"mode\", \"instance\", \"formulation\"]).size())\n"
        "results.head()",
    ),
    (
        "md",
        "## Figure 12 — penalty 균형과 feasible 비율\n\n"
        "빨간 점선은 coefficient range가 커지기 시작하는 지점입니다. "
        "**그 왼쪽에서 얻어진 개선은 range를 전혀 늘리지 않고 얻은 것**입니다.",
    ),
    (
        "code",
        "from IPython.display import Image\n\n"
        "path = BP.plot_ratio_vs_feasibility(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## Figure 13 — penalty 균형과 해의 품질\n\n"
        "feasible solution을 더 많이 찾는 것과 더 좋은 solution을 찾는 것은 "
        "다릅니다. feasible 비율이 올라가도 gap이 그대로면, 제약은 만족시키되 "
        "**그중 어느 것이 싼지는 구분하지 못한다**는 뜻입니다.",
    ),
    (
        "code",
        "path = BP.plot_ratio_vs_gap(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## Figure 14 — 제약 강도와 하드웨어 노이즈\n\n"
        "이 실험의 출발점이 된 그림입니다.\n\n"
        "`ratio = 1`(기존 설정)에서 assignment 강도가 capacity보다 몇 자릿수 "
        "아래인지, 그리고 어느 ratio에서 회색 띠(ICE 노이즈) 위로 올라오는지를 "
        "봅니다.\n\n"
        "objective 곡선이 계속 띠 아래에 머문다면, **제약은 보이게 만들 수 "
        "있어도 objective는 여전히 보이지 않는다**는 뜻입니다.",
    ),
    (
        "code",
        "path = BP.plot_constraint_strength(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## Figure 15 — penalty 균형과 coefficient range\n\n"
        "일정 ratio까지 range가 완전히 평평하다는 것을 보여줍니다. "
        "그 구간의 개선은 **dynamic range를 대가로 치르지 않았습니다.**",
    ),
    (
        "code",
        "path = BP.plot_ratio_vs_range(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## 요약\n\n"
        "아래 출력은 관측된 수치만 기술합니다. 오프라인 결과는 노이즈가 없는 "
        "이상적 조건이므로, QPU 결과의 **상한**으로 읽어야 합니다.",
    ),
    (
        "code",
        "lines = PB.summarize(results)\n"
        "for line in lines:\n"
        "    print(line)\n"
        "\n"
        "summary_path = PROCESSED_DIR / \"penalty_balance_summary.txt\"\n"
        "summary_path.write_text(\"\\n\".join(lines), encoding=\"utf-8\")\n"
        "print(\"\\n저장:\", summary_path)",
    ),
]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in {
        "12_lambda_balance_experiment.ipynb": NB12,
        "13_lambda_balance_figures.ipynb": NB13,
    }.items():
        path = NOTEBOOK_DIR / filename
        nbf.write(_notebook(cells), path)
        print(f"생성: {path.relative_to(PROJECT_ROOT)} ({len(cells)} 셀)")


if __name__ == "__main__":
    main()
