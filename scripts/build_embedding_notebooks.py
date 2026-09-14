"""notebooks/14, 15, 16 (embedding 실패 원인 분석)을 생성한다.

    14_embedding_timeout_sweep.ipynb : timeout sweep
    15_embedding_tries_sweep.ipynb   : tries sweep (timeout 고정)
    16_embedding_figures.ipynb       : 집계 및 Figure 16~19

실행:
    python scripts/build_embedding_notebooks.py
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
STUDY = config["embedding_study"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
print("설정 로드 완료")
'''

PURPOSE = (
    "## 실험 목적\n\n"
    "본 실험(notebook 05)에서 MS는 6x6 이상에서 embedding에 실패했습니다. "
    "원인 후보가 셋인데 관측만으로는 구분되지 않습니다.\n\n"
    "| 관측 | 원인 |\n"
    "|---|---|\n"
    "| timeout/tries를 늘렸더니 성공 | **(a) search budget** |\n"
    "| Pegasus 실패, Zephyr 성공 | **(b) topology / connectivity** |\n"
    "| 충분한 budget + Zephyr에서도 실패 | **(c) MS formulation 자체** |\n\n"
    "embedding 탐색은 확률적이므로 seed 하나로 판정하지 않고 여러 seed의 "
    "**성공률**로 봅니다.\n\n"
    "embedding 가능 여부는 QUBO의 **그래프 구조에만** 의존하고 penalty 값과는 "
    "무관합니다. 따라서 본 실험의 기본 설정으로 QUBO를 한 번만 만들어 씁니다."
)

SAVE_CELL = (
    "OUTPUT = PROCESSED_DIR / \"embedding_study.csv\"\n"
    "if OUTPUT.exists():\n"
    "    previous = pd.read_csv(OUTPUT)\n"
    "    key = [\"instance\", \"formulation\", \"topology\", \"sweep\",\n"
    "           \"timeout\", \"tries\", \"seed\"]\n"
    "    merged = pd.concat([previous, results], ignore_index=True)\n"
    "    merged = merged.drop_duplicates(subset=key, keep=\"last\")\n"
    "else:\n"
    "    merged = results\n"
    "save_table(merged, PROCESSED_DIR, \"embedding_study.csv\")\n"
    "print(f\"저장: {OUTPUT} ({len(merged)} 행)\")\n"
    "print(merged.groupby([\"sweep\", \"topology\"])[\"success\"].agg([\"count\", \"mean\"]))"
)


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


NB14 = [
    (
        "md",
        "# 14. Embedding timeout sweep\n\n" + PURPOSE + "\n\n"
        "이 notebook은 **tries를 고정하고 timeout을 훑습니다.** tries는 "
        "notebook 15에서 따로 다룹니다.",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        "## 설정\n\n"
        "- `QA_DRYRUN` — True면 이상적 topology 그래프, False면 실제 QPU의 "
        "working graph를 대상으로 한다. QPU는 결함 큐빗이 제외되어 있어 "
        "이상적 그래프보다 불리하고, 접속한 solver의 topology 하나만 쓸 수 있다.\n"
        "- 나머지 값은 `config/experiment_config.yaml`의 `embedding_study` "
        "섹션에서 관리한다. 여기서 덮어쓸 수도 있다.\n\n"
        "`noise` 설정은 config에 있지만 **embedding 탐색은 그래프 구조만 쓰므로 "
        "영향을 받지 않는다.** sampling을 함께 하는 실험에서만 쓰인다.",
    ),
    (
        "code",
        "QA_DRYRUN = True\n"
        "\n"
        "TARGETS = [\n"
        "    (\"4x4\", \"MS\"),\n"
        "    (\"6x6\", \"MS\"),\n"
        "    (\"8x8\", \"MS\"),\n"
        "    (\"15x15\", \"MS\"),\n"
        "]\n"
        "\n"
        "TOPOLOGIES = [tuple(item) for item in STUDY[\"topologies\"]]\n"
        "TIMEOUTS = list(STUDY[\"timeouts\"])\n"
        "SEEDS = list(STUDY[\"seeds\"])\n"
        "TRIES = int(STUDY[\"timeout_sweep_tries\"])\n"
        "\n"
        "print(f\"모드       : {'이상적 topology 그래프' if QA_DRYRUN else '실제 QPU working graph'}\")\n"
        "print(f\"대상       : {TARGETS}\")\n"
        "print(f\"topology   : {TOPOLOGIES}\")\n"
        "print(f\"timeout    : {TIMEOUTS}\")\n"
        "print(f\"seed       : {SEEDS}\")\n"
        "print(f\"tries(고정): {TRIES}\")\n"
        "print(f\"noise      : {config['noise']}  (embedding에는 영향 없음)\")",
    ),
    (
        "md",
        "## 실행 시간 추정\n\n"
        "**실패하는 경우는 timeout을 전부 소진합니다.** 따라서 최악의 경우 "
        "소요 시간은 아래와 같습니다. 큰 instance에서는 수 시간이 걸릴 수 "
        "있으므로 반드시 먼저 확인하십시오.\n\n"
        "시간이 부족하면 `TARGETS`를 줄이거나 `TIMEOUTS`의 큰 값을 빼십시오.",
    ),
    (
        "code",
        "from src import embedding_study as ES\n\n"
        "budget = ES.estimate_budget(\n"
        "    TARGETS, ES.normalize_topologies(TOPOLOGIES), TIMEOUTS, SEEDS\n"
        ")\n"
        "print(f\"총 실행 횟수      : {budget['runs']:.0f}\")\n"
        "print(f\"최악의 경우(전부 실패): {budget['worst_case_minutes']:.0f}분 \"\n"
        "      f\"({budget['worst_case_hours']:.1f}시간)\")\n"
        "print(\"성공하면 훨씬 짧게 끝납니다.\")",
    ),
    (
        "md",
        "## 사용 가능한 solver 확인 (online 모드일 때만)\n\n"
        "`QA_DRYRUN=False`이면 지정한 topology를 가진 solver를 골라 접속합니다. "
        "**Zephyr는 Advantage2 계열이므로 계정에 해당 solver가 없으면 "
        "사용할 수 없습니다.** 아래에서 먼저 확인하십시오.\n\n"
        "dry-run이면 이 셀은 건너뜁니다.",
    ),
    (
        "code",
        "from src import embedding_study as ES\n\n"
        "if QA_DRYRUN:\n"
        "    print(\"dry-run 모드입니다. 이상적 topology 그래프를 사용합니다.\")\n"
        "else:\n"
        "    try:\n"
        "        display(ES.list_available_solvers())\n"
        "    except Exception as error:\n"
        "        print(\"solver 목록을 가져오지 못했습니다:\", error)",
    ),
    (
        "md",
        "## topology 비교 기준\n\n"
        "두 topology의 규모와 연결도를 먼저 확인합니다. Zephyr가 노드도 많고 "
        "평균 차수도 높으므로, **Zephyr에서만 성공한다면 그것은 connectivity "
        "덕분**이라고 해석할 수 있습니다.",
    ),
    (
        "code",
        "import dwave_networkx as dnx\n\n"
        "rows = []\n"
        "for name, size in TOPOLOGIES:\n"
        "    builder = {\"pegasus\": dnx.pegasus_graph, \"zephyr\": dnx.zephyr_graph}[name]\n"
        "    graph = builder(size)\n"
        "    rows.append({\n"
        "        \"topology\": f\"{name}{size}\",\n"
        "        \"nodes\": graph.number_of_nodes(),\n"
        "        \"edges\": graph.number_of_edges(),\n"
        "        \"avg_degree\": round(\n"
        "            2 * graph.number_of_edges() / graph.number_of_nodes(), 1\n"
        "        ),\n"
        "    })\n"
        "pd.DataFrame(rows)",
    ),
    (
        "md",
        "## 대상 QUBO의 규모\n\n"
        "embedding 난이도는 변수 수보다 **엣지 수(이차항)** 에 좌우됩니다. "
        "MS는 `q_ij`를 binary expansion하면서 한 constraint 안의 항이 늘고, "
        "제곱 전개 시 이차항이 항 수의 제곱으로 늘어납니다.",
    ),
    (
        "code",
        "from src.data_generator import CFLPInstance\n\n"
        "rows = []\n"
        "for name, formulation in TARGETS:\n"
        "    instance = CFLPInstance.load(DATA_DIR / f\"{name}.json\")\n"
        "    _, variables, edges = ES._source_graph(instance, formulation, config)\n"
        "    rows.append({\n"
        "        \"instance\": name,\n"
        "        \"formulation\": formulation,\n"
        "        \"logical_variables\": variables,\n"
        "        \"logical_edges\": edges,\n"
        "        \"density\": round(2 * edges / (variables * (variables - 1)), 4),\n"
        "    })\n"
        "pd.DataFrame(rows)",
    ),
    ("md", "## timeout sweep 실행"),
    (
        "code",
        "results = ES.run_timeout_sweep(\n"
        "    targets=TARGETS,\n"
        "    config=config,\n"
        "    data_dir=DATA_DIR,\n"
        "    timeouts=TIMEOUTS,\n"
        "    seeds=SEEDS,\n"
        "    tries=TRIES,\n"
        "    topologies=TOPOLOGIES,\n"
        "    qa_dryrun=QA_DRYRUN,\n"
        ")\n"
        "print(f\"\\n{len(results)} 회 실행 완료\")",
    ),
    (
        "md",
        "## 결과 저장\n\n"
        "timeout sweep과 tries sweep을 **같은 파일에 누적**합니다. "
        "notebook 15를 실행하면 이어서 쌓이고, notebook 16에서 함께 집계합니다.",
    ),
    ("code", "from src.persistence import save_table\n\n" + SAVE_CELL),
    (
        "md",
        "## 중간 판정\n\n"
        "timeout을 늘렸을 때 성공률이 오르면 **search budget 문제**입니다. "
        "모든 timeout에서 0%라면 budget이 아니라 topology나 formulation 쪽을 "
        "봐야 합니다.",
    ),
    (
        "code",
        "pivot = results.pivot_table(\n"
        "    index=[\"instance\", \"logical_variables\", \"logical_edges\"],\n"
        "    columns=[\"topology\", \"timeout\"],\n"
        "    values=\"success\",\n"
        "    aggfunc=\"mean\",\n"
        ")\n"
        "(pivot * 100).round(0)",
    ),
]

NB15 = [
    (
        "md",
        "# 15. Embedding tries sweep\n\n" + PURPOSE + "\n\n"
        "이 notebook은 **timeout을 고정하고 tries를 훑습니다.**\n\n"
        "`tries`는 minorminer가 서로 다른 시작점에서 재시도하는 횟수입니다. "
        "timeout이 전체 시간 예산이라면 tries는 **탐색의 다양성**에 해당합니다. "
        "둘은 서로 다른 종류의 budget이므로 나누어 봅니다.\n\n"
        "고정할 timeout은 notebook 14의 결과를 보고 정하십시오. 성공률이 "
        "포화되기 시작하는 값이 적절합니다.",
    ),
    ("code", BOOTSTRAP),
    (
        "code",
        "QA_DRYRUN = True\n"
        "\n"
        "TARGETS = [\n"
        "    (\"4x4\", \"MS\"),\n"
        "    (\"6x6\", \"MS\"),\n"
        "    (\"8x8\", \"MS\"),\n"
        "    (\"15x15\", \"MS\"),\n"
        "]\n"
        "\n"
        "TOPOLOGIES = [tuple(item) for item in STUDY[\"topologies\"]]\n"
        "TRIES_GRID = list(STUDY[\"tries_grid\"])\n"
        "SEEDS = list(STUDY[\"seeds\"])\n"
        "FIXED_TIMEOUT = int(STUDY[\"tries_sweep_timeout\"])\n"
        "\n"
        "print(f\"모드          : {'이상적 topology 그래프' if QA_DRYRUN else '실제 QPU working graph'}\")\n"
        "print(f\"tries grid    : {TRIES_GRID}\")\n"
        "print(f\"timeout(고정) : {FIXED_TIMEOUT}s\")\n"
        "print(f\"seed          : {SEEDS}\")",
    ),
    (
        "md",
        "## 실행 시간 추정\n\n"
        "tries를 늘리면 minorminer가 재시도를 더 많이 하지만, **timeout이 전체 "
        "상한**이므로 한 번의 실행이 `FIXED_TIMEOUT`을 넘지는 않습니다. "
        "따라서 최악의 경우는 아래와 같습니다.",
    ),
    (
        "code",
        "from src import embedding_study as ES\n\n"
        "budget = ES.estimate_budget(\n"
        "    TARGETS, TOPOLOGIES, [FIXED_TIMEOUT] * len(TRIES_GRID), SEEDS\n"
        ")\n"
        "print(f\"총 실행 횟수      : {budget['runs']:.0f}\")\n"
        "print(f\"최악의 경우(전부 실패): {budget['worst_case_minutes']:.0f}분 \"\n"
        "      f\"({budget['worst_case_hours']:.1f}시간)\")",
    ),
    ("md", "## tries sweep 실행"),
    (
        "code",
        "results = ES.run_tries_sweep(\n"
        "    targets=TARGETS,\n"
        "    config=config,\n"
        "    data_dir=DATA_DIR,\n"
        "    tries_grid=TRIES_GRID,\n"
        "    seeds=SEEDS,\n"
        "    timeout=FIXED_TIMEOUT,\n"
        "    topologies=TOPOLOGIES,\n"
        "    qa_dryrun=QA_DRYRUN,\n"
        ")\n"
        "print(f\"\\n{len(results)} 회 실행 완료\")",
    ),
    ("code", "from src.persistence import save_table\n\n" + SAVE_CELL),
    (
        "code",
        "pivot = results.pivot_table(\n"
        "    index=[\"instance\", \"logical_variables\"],\n"
        "    columns=[\"topology\", \"tries\"],\n"
        "    values=\"success\",\n"
        "    aggfunc=\"mean\",\n"
        ")\n"
        "(pivot * 100).round(0)",
    ),
]

NB16 = [
    (
        "md",
        "# 16. Embedding 실험 집계 및 Figure\n\n"
        "notebook 14, 15의 결과를 합쳐 Figure 16~19와 요약 CSV를 만듭니다.\n\n"
        "| Figure | 내용 |\n"
        "|---|---|\n"
        "| 16 | timeout 대 embedding 성공률 (topology 비교, instance별 panel) |\n"
        "| 17 | 성공한 embedding의 품질 (physical qubits, avg/max chain length) |\n"
        "| 18 | tries 대 embedding 성공률 (timeout 고정) |\n"
        "| 19 | QUBO 엣지 수와 성공률의 관계 |",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        "## 결과 로드\n\n"
        "`embedding_study.csv`는 notebook 14, 15의 결과가 **누적**된 파일입니다. "
        "이전에 다른 조건으로 돌린 행이 남아 있을 수 있으므로, 무엇이 들어 "
        "있는지 먼저 확인하십시오.\n\n"
        "특정 topology만 보고 싶으면 `TOPOLOGY_FILTER`를 지정하십시오. "
        "`None`이면 전부 사용합니다.",
    ),
    (
        "code",
        "from src.persistence import load_table, save_table\n"
        "from src import embedding_study as ES\n"
        "from src import embedding_plotting as EP\n\n"
        "TOPOLOGY_FILTER = None   # 예: [\"zephyr\"] 또는 [\"pegasus\", \"zephyr\"]\n"
        "\n"
        "results = load_table(PROCESSED_DIR, \"embedding_study.csv\")\n"
        "print(\"파일에 들어 있는 조건:\")\n"
        "print(\n"
        "    results.groupby([\"sweep\", \"topology\", \"instance\"])[\"success\"]\n"
        "    .agg([\"count\", \"mean\"])\n"
        ")\n"
        "\n"
        "if TOPOLOGY_FILTER is not None:\n"
        "    before = len(results)\n"
        "    results = results[results[\"topology\"].isin(TOPOLOGY_FILTER)]\n"
        "    print(f\"\\n필터 적용: {before} -> {len(results)} 행\")\n"
        "results.head()",
    ),
    (
        "md",
        "## 요약 CSV\n\n"
        "`min_timeout_success`는 embedding이 처음 성공한 최소 timeout입니다. "
        "값이 있으면 **search budget으로 넘을 수 있는 벽**이었다는 뜻이고, "
        "`NaN`이면 모든 timeout에서 실패했다는 뜻입니다.",
    ),
    (
        "code",
        "summary = ES.summarize(results)\n"
        "save_table(summary, PROCESSED_DIR, \"embedding_study_summary.csv\")\n"
        "print(\"저장:\", PROCESSED_DIR / \"embedding_study_summary.csv\")\n"
        "summary",
    ),
    ("md", "## Figure 16 — timeout 대 성공률"),
    (
        "code",
        "from IPython.display import Image\n\n"
        "path = EP.plot_timeout_success(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## Figure 17 — 성공한 embedding의 품질\n\n"
        "성공했더라도 chain이 길면 chain break 확률이 높아 실제 solution "
        "quality가 나빠집니다. **성공/실패만이 아니라 품질도 함께 봐야** "
        "topology의 이점을 정확히 평가할 수 있습니다.",
    ),
    (
        "code",
        "path = EP.plot_embedding_quality(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    ("md", "## Figure 18 — tries 대 성공률"),
    (
        "code",
        "path = EP.plot_tries_success(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## Figure 19 — 그래프 규모와 성공률\n\n"
        "가로축이 변수 수가 아니라 **이차항 수**입니다. 점의 크기가 변수 수를 "
        "나타냅니다. 변수가 적은데도 실패하는 점이 있다면, 성공을 가르는 것이 "
        "변수 수가 아니라 그래프의 조밀도라는 뜻입니다.",
    ),
    (
        "code",
        "path = EP.plot_density_vs_success(results, FIGURE_DIR)\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## 원인 판정\n\n"
        "instance별로 세 원인 중 어느 것인지 자동 판정합니다. 자동 판정은 "
        "보조 지표이므로, 위 그림과 요약표를 함께 보고 최종 판단하십시오.",
    ),
    (
        "code",
        "lines = ES.diagnose(results)\n"
        "for line in lines:\n"
        "    print(line)\n"
        "\n"
        "summary_path = PROCESSED_DIR / \"embedding_study_summary.txt\"\n"
        "summary_path.write_text(\"\\n\".join(lines), encoding=\"utf-8\")\n"
        "print(\"\\n저장:\", summary_path)",
    ),
]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in {
        "14_embedding_timeout_sweep.ipynb": NB14,
        "15_embedding_tries_sweep.ipynb": NB15,
        "16_embedding_figures.ipynb": NB16,
    }.items():
        path = NOTEBOOK_DIR / filename
        nbf.write(_notebook(cells), path)
        print(f"생성: {path.relative_to(PROJECT_ROOT)} ({len(cells)} 셀)")


if __name__ == "__main__":
    main()
