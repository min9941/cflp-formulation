"""notebooks/10, 11 (penalty coefficient 민감도 실험)을 생성한다.

    10_lambda_experiment.ipynb  : lambda grid 실행 및 결과 저장
    11_lambda_figures.ipynb     : Figure 8~11 생성 및 해석

실행:
    python scripts/build_lambda_notebooks.py
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
SOLUTION_DIR = RAW_DIR / "solutions"

pd.set_option("display.width", 180)
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


NB10 = [
    (
        "md",
        "# 10. Penalty coefficient(lambda) 민감도 실험\n\n"
        "## 왜 이 실험을 하는가\n\n"
        "본 실험(notebook 04, 05)에서는 lambda를 시작 전에 고정하고 결과를 보고 "
        "조정하지 않았습니다. 그 결과 QA는 24회 중 1회만 feasible solution을 냈고, "
        "QUBO 계수 범위가 10^9에 달했습니다.\n\n"
        "이 notebook은 **본 실험의 일부가 아니라 그 결과를 진단하기 위한 별도 "
        "실험**입니다. lambda를 의도적으로 바꿔가며 두 가설을 구분합니다.\n\n"
        "| 가설 | 내용 | 예측 |\n"
        "|---|---|---|\n"
        "| A. precision | lambda가 커서 계수 범위가 넓어지고, objective 정보가 "
        "하드웨어의 아날로그 정밀도 아래로 묻힌다 | lambda를 낮추면 **QA만** 개선 |\n"
        "| B. landscape | lambda가 커서 penalty 지형이 험해지고, solver가 지형 "
        "자체를 탐색하지 못한다 | lambda를 낮추면 **SA와 QA가 함께** 개선 |\n\n"
        "SA는 소프트웨어라 아날로그 정밀도 제약이 없습니다. 따라서 **SA를 "
        "classical control로 두면 두 가설이 구분됩니다.** 이것이 이 실험 설계의 "
        "핵심입니다.\n\n"
        "## 대상\n\n"
        "**SS 8x8**을 주 대상으로 합니다. 이유는 세 가지입니다.\n\n"
        "- embedding에 성공했습니다. 실패하는 조합에서는 lambda를 바꿔도 관측할 "
        "것이 없습니다.\n"
        "- 그럼에도 QA feasible 비율이 0%였습니다. 개선 여지가 관측 가능합니다.\n"
        "- QUBO 변수가 117개로 QPU 시간을 과하게 쓰지 않습니다.\n\n"
        "여유가 있으면 SS 4x4(QA가 유일하게 성공한 경우)와 MS 4x4(embedding은 "
        "됐으나 feasible 0)를 추가로 돌려 비교하십시오.",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        "## lambda grid 설계\n\n"
        "lambda의 이론적 하한은 U_obj가 아니라 **Z_ub**입니다.\n\n"
        "infeasible solution의 energy는 objective가 0 이상이므로 최소 lambda입니다. "
        "한편 optimal solution의 energy Z*는 알려진 feasible solution의 objective "
        "Z_ub 이하입니다. 따라서\n\n"
        "$$\\lambda > Z_{ub} \\implies \\text{ground state가 feasible}$$\n\n"
        "이 성립합니다. 현재 설정은 `lambda = 1.1 x U_obj`인데, SS 8x8에서 이 값은 "
        "Z_ub의 약 4.56배입니다. 즉 필요한 것보다 4.5배 큽니다.\n\n"
        "그래서 grid를 `m = lambda / Z_ub`로 두고 **m = 1이 이론적 경계**가 되도록 "
        "했습니다.\n\n"
        "- `m < 1` : 이론적 보장이 깨지는 영역. **진단 목적으로만** 사용하고 본 "
        "실험 결과로는 보고하지 않습니다.\n"
        "- `m = 1` : 이론적 경계\n"
        "- `m = 4.56` : 현재 설정\n",
    ),
    (
        "code",
        "from src.data_generator import CFLPInstance\n"
        "from src.persistence import load_solution, save_table\n"
        "from src.decoder import encode_solution\n"
        "from src.qubo_builder import build_qubo\n"
        "from src import lambda_experiment as LE\n\n"
        "TARGET_INSTANCE = \"8x8\"\n"
        "TARGET_FORMULATION = \"SS\"\n\n"
        "instance = CFLPInstance.load(DATA_DIR / f\"{TARGET_INSTANCE}.json\")\n"
        "gurobi = pd.read_csv(RAW_DIR / \"gurobi_results.csv\")\n"
        "model_name = \"SS\" if TARGET_FORMULATION == \"SS\" else \"MS\"\n"
        "reference_objective = float(\n"
        "    gurobi[(gurobi[\"instance\"] == TARGET_INSTANCE)\n"
        "           & (gurobi[\"gurobi_model\"] == model_name)][\"objective\"].iloc[0]\n"
        ")\n"
        "reference_solution = load_solution(\n"
        "    SOLUTION_DIR, TARGET_INSTANCE, f\"gurobi_{model_name.replace('-', '_')}\"\n"
        ")\n"
        "base_model = build_qubo(instance, TARGET_FORMULATION, float(config[\"penalty\"][\"margin\"]))\n"
        "reference_sample = encode_solution(base_model, instance, reference_solution)\n\n"
        "current_multiplier = base_model.penalty.value / reference_objective\n"
        "print(f\"Z_ub (Gurobi optimum) = {reference_objective:.2f}\")\n"
        "print(f\"현재 lambda           = {base_model.penalty.value:.2f}\")\n"
        "print(f\"현재 m = lambda/Z_ub  = {current_multiplier:.2f}\")\n"
        "print(f\"grid                  = {LE.DEFAULT_MULTIPLIERS}\")",
    ),
    (
        "md",
        "## SA sweep (classical control)\n\n"
        "SA는 아날로그 정밀도 제약이 없으므로, SA가 lambda에 어떻게 반응하는지가 "
        "**가설 B의 크기를 직접 알려줍니다.**\n\n"
        "SA 파라미터는 본 실험과 동일하게 유지합니다. 바꾸는 것은 lambda 하나뿐입니다.",
    ),
    (
        "code",
        "sa_points = LE.run_sa_sweep(\n"
        "    instance=instance,\n"
        "    formulation=TARGET_FORMULATION,\n"
        "    reference_objective=reference_objective,\n"
        "    reference_sample=reference_sample,\n"
        "    sa_config=config[\"sa\"],\n"
        "    tolerance=float(config[\"feasibility\"][\"tolerance\"]),\n"
        ")\n"
        "sa_frame = pd.DataFrame([point.to_record() for point in sa_points])\n"
        "sa_frame.insert(0, \"instance\", TARGET_INSTANCE)\n"
        "sa_frame.insert(1, \"formulation\", TARGET_FORMULATION)\n"
        "sa_frame[[\n"
        "    \"multiplier\", \"lambda_value\", \"qubo_range\",\n"
        "    \"scaled_objective_coefficient\", \"feasible_fraction\",\n"
        "    \"true_gap_percent\", \"ground_state_is_feasible\",\n"
        "]]",
    ),
    (
        "md",
        "## 하드웨어 정밀도와의 비교\n\n"
        "`scaled_objective_coefficient`는 D-Wave의 auto_scale 이후 objective 계수가 "
        "얼마나 작아지는지를 나타냅니다. D-Wave는 모든 계수를 max|J|로 나누어 "
        "하드웨어 범위에 맞추기 때문입니다.\n\n"
        "ICE(integrated control error)에 의한 노이즈는 정규화 단위로 대략 "
        "0.01~0.03 수준으로 알려져 있습니다. objective 계수가 이보다 작으면 "
        "**하드웨어가 objective를 구분할 수 없다**는 뜻입니다.",
    ),
    (
        "code",
        "ICE_NOISE = 0.02   # 정규화 단위. 정확한 값이 아니라 자릿수 비교용 기준.\n"
        "view = sa_frame[[\"multiplier\", \"lambda_value\", \"scaled_objective_coefficient\"]].copy()\n"
        "view[\"noise_ratio\"] = view[\"scaled_objective_coefficient\"] / ICE_NOISE\n"
        "view[\"objective_visible\"] = view[\"noise_ratio\"] > 1.0\n"
        "print(\"objective 계수가 ICE 노이즈보다 큰 lambda가 하나라도 있는가:\",\n"
        "      bool(view[\"objective_visible\"].any()))\n"
        "view",
    ),
    (
        "md",
        "## QA sweep\n\n"
        "**중요**: QUBO의 그래프 구조는 lambda와 무관합니다. lambda는 계수의 크기만 "
        "바꿀 뿐 어떤 변수쌍이 연결되는지는 바꾸지 않습니다. 따라서 embedding을 "
        "**한 번만 계산해 모든 lambda에 재사용**합니다.\n\n"
        "이렇게 해야 관측된 차이가 embedding 운이 아니라 lambda 때문임을 보장할 수 "
        "있습니다.\n\n"
        "chain strength는 `alpha x max|J|` 규칙이므로 lambda에 따라 자동으로 함께 "
        "움직입니다. 즉 chain의 상대적 강도는 일정하게 유지됩니다.\n\n"
        "토큰이 없으면 이 셀은 건너뛰고 SA 결과만으로 진행합니다.",
    ),
    (
        "code",
        "import os\n\n"
        "qa_frame = pd.DataFrame()\n"
        "if not os.environ.get(\"DWAVE_API_TOKEN\"):\n"
        "    print(\"DWAVE_API_TOKEN이 없습니다. QA sweep을 건너뜁니다.\")\n"
        "    print(\"토큰을 설정한 뒤 이 셀만 다시 실행하면 됩니다.\")\n"
        "else:\n"
        "    qa_points = LE.run_qa_sweep(\n"
        "        instance=instance,\n"
        "        formulation=TARGET_FORMULATION,\n"
        "        reference_objective=reference_objective,\n"
        "        reference_sample=reference_sample,\n"
        "        qa_config=config[\"qa\"],\n"
        "        embedding_config=config[\"embedding\"],\n"
        "        tolerance=float(config[\"feasibility\"][\"tolerance\"]),\n"
        "    )\n"
        "    qa_frame = pd.DataFrame([point.to_record() for point in qa_points])\n"
        "    qa_frame.insert(0, \"instance\", TARGET_INSTANCE)\n"
        "    qa_frame.insert(1, \"formulation\", TARGET_FORMULATION)\n"
        "    display(qa_frame[[\n"
        "        \"multiplier\", \"lambda_value\", \"feasible_fraction\",\n"
        "        \"true_gap_percent\", \"qa_chain_break_fraction\",\n"
        "    ]])",
    ),
    (
        "code",
        "lambda_results = pd.concat([sa_frame, qa_frame], ignore_index=True)\n"
        "save_table(lambda_results, PROCESSED_DIR, \"lambda_sweep.csv\")\n"
        "print(\"저장:\", PROCESSED_DIR / \"lambda_sweep.csv\", f\"({len(lambda_results)} 행)\")",
    ),
    (
        "md",
        "## 가설 판정\n\n"
        "lambda가 낮은 구간(m <= 1)과 높은 구간(m >= 3)의 feasible 비율을 비교합니다.",
    ),
    (
        "code",
        "for line in LE.interpret(lambda_results):\n"
        "    print(line)",
    ),
]

NB11 = [
    (
        "md",
        "# 11. Penalty coefficient 실험 결과 시각화\n\n"
        "notebook 10의 결과로 Figure 8~11을 만듭니다.\n\n"
        "| Figure | 내용 | 확인할 것 |\n"
        "|---|---|---|\n"
        "| 8 | lambda 대 QUBO 계수 범위 | lambda에 정확히 비례하는가 |\n"
        "| 9 | lambda 대 feasible 비율 (SA vs QA) | **두 가설을 구분하는 핵심 그림** |\n"
        "| 10 | lambda 대 true optimality gap | 품질도 함께 개선되는가 |\n"
        "| 11 | 정규화된 objective 계수와 ICE 노이즈 | 어느 lambda에서 objective가 보이는가 |",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "from src.persistence import load_table\n"
        "from src import lambda_plotting as LP\n\n"
        "results = load_table(PROCESSED_DIR, \"lambda_sweep.csv\")\n"
        "print(\"solver별 행 수:\")\n"
        "print(results.groupby(\"solver\").size())\n"
        "results.head()",
    ),
    (
        "md",
        "## Figure 8 — lambda와 계수 범위\n\n"
        "penalty term이 제곱 형태이므로 계수는 `lambda x (제약계수)^2` 규모가 "
        "됩니다. 제약계수는 lambda와 무관하므로 **계수 범위는 lambda에 정확히 "
        "비례**해야 합니다. 그래프가 직선(log-log)이면 구현이 의도대로 된 것입니다.",
    ),
    (
        "code",
        "path = LP.plot_lambda_vs_range(results, FIGURE_DIR)\n"
        "from IPython.display import Image\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## Figure 9 — lambda와 feasible 비율 (핵심)\n\n"
        "이 그림 하나가 두 가설을 구분합니다.\n\n"
        "- **QA 곡선만 왼쪽에서 올라가면** → 가설 A(precision)\n"
        "- **SA와 QA가 함께 올라가면** → 가설 B(landscape)의 기여가 큼\n"
        "- **둘 다 평평하면** → lambda 외의 요인\n\n"
        "세로 점선은 이론적 경계(m = 1)입니다. 그 왼쪽은 ground state가 feasible "
        "이라는 보장이 없는 영역이므로, 결과가 좋아 보여도 **해로 채택할 수 "
        "없습니다.** 진단용으로만 읽으십시오.",
    ),
    (
        "code",
        "path = LP.plot_lambda_vs_feasibility(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## Figure 10 — lambda와 optimality gap\n\n"
        "feasible solution을 찾는 것과 좋은 solution을 찾는 것은 다릅니다. "
        "feasible 비율이 올라가도 gap이 그대로면, penalty를 낮춘 효과가 "
        "**품질까지 이어지지는 않는다**는 뜻입니다.",
    ),
    (
        "code",
        "path = LP.plot_lambda_vs_gap(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## Figure 11 — objective 계수와 하드웨어 노이즈\n\n"
        "D-Wave는 모든 계수를 max|J|로 나누어 하드웨어 범위에 맞춥니다. 그 결과 "
        "원래 objective 계수가 얼마나 작아지는지를 ICE 노이즈 수준과 비교합니다.\n\n"
        "회색 띠 아래에 있으면 objective가 노이즈에 묻혀 하드웨어가 구분할 수 "
        "없다는 뜻입니다. **이 그림에서 모든 점이 띠 아래에 있다면, lambda 조정 "
        "만으로는 QA를 살릴 수 없다는 정량적 근거가 됩니다.**",
    ),
    (
        "code",
        "path = LP.plot_scaled_objective(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## 요약\n\n"
        "아래 출력은 관측된 수치만 기술합니다. lambda를 낮춰 결과가 좋아지더라도, "
        "m < 1 구간은 이론적 보장이 없으므로 본 실험의 설정으로 채택할 수 "
        "없습니다.",
    ),
    (
        "code",
        "from src import lambda_experiment as LE\n\n"
        "lines = LE.interpret(results)\n"
        "for line in lines:\n"
        "    print(line)\n"
        "\n"
        "summary_path = PROCESSED_DIR / \"lambda_summary.txt\"\n"
        "summary_path.write_text(\"\\n\".join(lines), encoding=\"utf-8\")\n"
        "print(\"\\n저장:\", summary_path)",
    ),
]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in {
        "10_lambda_experiment.ipynb": NB10,
        "11_lambda_figures.ipynb": NB11,
    }.items():
        path = NOTEBOOK_DIR / filename
        nbf.write(_notebook(cells), path)
        print(f"생성: {path.relative_to(PROJECT_ROOT)} ({len(cells)} 셀)")


if __name__ == "__main__":
    main()
