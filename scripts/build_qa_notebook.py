"""notebooks/30_MS_6x6_fixed_embedding_qa.ipynb 를 생성한다.

실행:
    python scripts/build_qa_notebook.py
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
EMBEDDING_DIR = RAW_DIR / "embeddings"
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
print("설정 로드 완료")
'''

CELLS: list[tuple[str, str]] = [
    (
        "md",
        "# 30. MS 6x6 고정 embedding QA 실행\n\n"
        "## 배경\n\n"
        "본 실험(notebook 05)에서 MS 6x6은 timeout이 부족해 embedding에 실패했고 "
        "`NOT_EMBEDDABLE`로 기록되었습니다. 이후 embedding 실험(notebook 14)에서 "
        "**Pegasus, tries=10, timeout=2400** 조건으로 5개 seed 전부 성공했습니다.\n\n"
        "| seed | 물리 큐빗 | 평균 chain | 최대 chain |\n"
        "|---|---|---|---|\n"
        "| 2024 | 4,503 | 17.39 | 34 |\n"
        "| 2026 | 4,537 | 17.52 | **31** |\n"
        "| 2025 | 4,675 | 18.05 | 35 |\n"
        "| 2028 | 4,705 | 18.17 | 40 |\n"
        "| 2027 | 4,755 | 18.36 | 37 |\n\n"
        "## 이 notebook의 목적\n\n"
        "1. notebook 05에서 비어 있던 MS 6x6 칸을 채웁니다.\n"
        "2. **chain break의 기여와 coefficient range의 기여를 분리**합니다.\n\n"
        "두 번째가 핵심입니다. MS 4x4는 embedding도 되고 chain break도 0.025%로 "
        "매우 낮았는데 feasible 해가 **0개**였습니다. 그래서 원인을 coefficient "
        "range로 추정했는데, 확증은 아니었습니다.\n\n"
        "6x6은 최대 chain이 31~40으로 4x4(최대 14~17)보다 훨씬 깁니다. 두 경우를 "
        "나란히 놓으면 어느 요인이 지배적인지 가늠할 수 있습니다.\n\n"
        "| 관측 | 해석 |\n"
        "|---|---|\n"
        "| 6x6도 chain break가 낮은데 feasible 0 | coefficient range가 지배적 |\n"
        "| 6x6에서 chain break가 급증 | chain 길이도 유의한 요인 |",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        "## 설정\n\n"
        "`CHAIN_STRENGTH_ALPHA`는 본 실험의 고정값 **1.5**입니다. "
        "`chain_strength = alpha * max|J|` 규칙의 alpha이고, 이 값을 바꾸는 것은 "
        "parameter tuning에 해당하므로 **먼저 1.5로 한 번 돌린 뒤** 결과를 보고 "
        "판단하십시오. 바꿔서 돌린 결과는 본 실험이 아니라 진단 실험으로 "
        "분류해 기록해야 합니다.\n\n"
        "`SELECT_METRICS`는 어떤 지표로 seed를 고를지입니다. 세 지표가 서로 다른 "
        "seed를 가리킬 수 있으므로(2024는 큐빗 최소, 2026은 최대 chain 최소), "
        "각 지표의 최선을 모두 뽑은 뒤 중복을 제거합니다.",
    ),
    (
        "code",
        "TARGET_INSTANCE = \"6x6\"\n"
        "TARGET_FORMULATION = \"MS\"\n"
        "TARGET_TOPOLOGY = \"pegasus\"\n"
        "\n"
        "# embedding 탐색 조건 (notebook 14에서 성공했던 조건과 동일해야 함)\n"
        "EMBED_TRIES = 10\n"
        "EMBED_TIMEOUT = 2400\n"
        "\n"
        "# seed 선택\n"
        "SELECT_METRICS = (\"physical_qubits\", \"avg_chain_length\", \"max_chain_length\")\n"
        "MAX_SEEDS = None          # None이면 선정된 전부\n"
        "\n"
        "# QA 파라미터. alpha는 본 실험 고정값 1.5로 먼저 돌린다.\n"
        "CHAIN_STRENGTH_ALPHA = 1.5\n"
        "NUM_READS = None          # None이면 config의 qa.num_reads\n"
        "ANNEALING_TIME = None     # None이면 config의 qa.annealing_time\n"
        "\n"
        "print(f\"대상      : {TARGET_INSTANCE} {TARGET_FORMULATION} / {TARGET_TOPOLOGY}\")\n"
        "print(f\"embedding : tries={EMBED_TRIES}, timeout={EMBED_TIMEOUT}s\")\n"
        "print(f\"alpha     : {CHAIN_STRENGTH_ALPHA}  (본 실험 고정값)\")\n"
        "print(f\"num_reads : {NUM_READS or config['qa']['num_reads']}\")",
    ),
    (
        "md",
        "## seed 선택\n\n"
        "embedding 실험 결과에서 품질이 좋은 seed를 고릅니다. 물리 큐빗이 가장 "
        "적은 embedding과 최대 chain이 가장 짧은 embedding이 다를 때 어느 쪽이 "
        "QA에 유리한지는 미리 알 수 없으므로, **둘 다 돌려서 비교**합니다.\n\n"
        "이것 자체가 \"embedding 품질이 QA 결과에 영향을 주는가\"라는 별도 질문에 "
        "대한 대조 실험이 됩니다.",
    ),
    (
        "code",
        "from src import fixed_embedding_qa as FQ\n"
        "from src.persistence import load_table\n\n"
        "embedding_results = load_table(PROCESSED_DIR, \"embedding_study.csv\")\n"
        "selected = FQ.select_seeds(\n"
        "    embedding_results,\n"
        "    instance=TARGET_INSTANCE,\n"
        "    formulation=TARGET_FORMULATION,\n"
        "    topology=TARGET_TOPOLOGY,\n"
        "    metrics=SELECT_METRICS,\n"
        "    max_seeds=MAX_SEEDS,\n"
        ")\n"
        "selected[[\n"
        "    \"seed\", \"selected_by\", \"physical_qubits\",\n"
        "    \"avg_chain_length\", \"max_chain_length\", \"tries\", \"timeout\",\n"
        "]]",
    ),
    (
        "md",
        "## embedding 확보\n\n"
        "mapping이 JSON으로 저장돼 있으면 재사용하고, 없으면 탐색해서 저장합니다.\n\n"
        "**저장본을 쓰기 전에 현재 working graph에서 유효한지 검사합니다.** QPU의 "
        "working graph는 교정 과정에서 바뀔 수 있고, 그러면 chain에 쓰인 물리 "
        "큐빗이 사라지거나 연결이 끊겨 mapping을 쓸 수 없게 됩니다. 이 경우 "
        "자동으로 재탐색합니다.\n\n"
        "재탐색은 seed당 수백~1,400초가 걸릴 수 있습니다.",
    ),
    (
        "code",
        "from src.data_generator import CFLPInstance\n\n"
        "instance = CFLPInstance.load(DATA_DIR / f\"{TARGET_INSTANCE}.json\")\n"
        "\n"
        "embeddings = {}\n"
        "for _, row in selected.iterrows():\n"
        "    seed = int(row[\"seed\"])\n"
        "    print(f\"--- seed {seed} ({row['selected_by']}) ---\")\n"
        "    embedding, meta = FQ.obtain_embedding(\n"
        "        instance=instance,\n"
        "        formulation=TARGET_FORMULATION,\n"
        "        config=config,\n"
        "        seed=seed,\n"
        "        tries=EMBED_TRIES,\n"
        "        timeout=EMBED_TIMEOUT,\n"
        "        topology=TARGET_TOPOLOGY,\n"
        "        embedding_dir=EMBEDDING_DIR,\n"
        "    )\n"
        "    embeddings[seed] = (embedding, meta)\n"
        "print(f\"\\n{len(embeddings)}개 embedding 확보\")",
    ),
    (
        "md",
        "## QUBO 생성\n\n"
        "본 실험과 동일한 설정으로 만듭니다. penalty도 기본값 그대로입니다.",
    ),
    (
        "code",
        "from src.qubo_builder import build_qubo, qubo_statistics\n\n"
        "model = build_qubo(\n"
        "    instance,\n"
        "    TARGET_FORMULATION,\n"
        "    float(config[\"penalty\"][\"margin\"]),\n"
        "    int(config[\"encoding\"][\"precision\"]),\n"
        ")\n"
        "stats = qubo_statistics(model)\n"
        "print(f\"변수 {stats['qubo_variables']}, 이차항 {stats['qubo_quadratic_terms']}\")\n"
        "print(f\"lambda {stats['penalty_lambda']:.4e}\")\n"
        "print(f\"계수 범위 {stats['qubo_range']:.4e}\")\n"
        "\n"
        "gurobi = pd.read_csv(RAW_DIR / \"gurobi_results.csv\")\n"
        "reference = float(\n"
        "    gurobi[(gurobi[\"instance\"] == TARGET_INSTANCE)\n"
        "           & (gurobi[\"gurobi_model\"] == TARGET_FORMULATION)][\"objective\"].iloc[0]\n"
        ")\n"
        "print(f\"Gurobi optimum {reference:.2f}\")",
    ),
    (
        "md",
        "## QA 실행\n\n"
        "선정된 embedding 각각에 대해 QA를 돌립니다. QUBO와 alpha는 동일하고 "
        "**embedding만 다릅니다.** 따라서 결과 차이는 embedding 품질에서 옵니다.",
    ),
    (
        "code",
        "records = []\n"
        "for seed, (embedding, meta) in embeddings.items():\n"
        "    print(f\"--- seed {seed} QA 실행 ---\")\n"
        "    outcome = FQ.run_qa_with_embedding(\n"
        "        model=model,\n"
        "        instance=instance,\n"
        "        embedding=embedding,\n"
        "        config=config,\n"
        "        chain_strength_alpha=CHAIN_STRENGTH_ALPHA,\n"
        "        solver_id=meta.get(\"solver_id\"),\n"
        "        num_reads=NUM_READS,\n"
        "        annealing_time=ANNEALING_TIME,\n"
        "    )\n"
        "    row = FQ.to_schema_row(\n"
        "        outcome=outcome,\n"
        "        meta=meta,\n"
        "        model=model,\n"
        "        instance=instance,\n"
        "        formulation=TARGET_FORMULATION,\n"
        "        selected_by=selected[selected[\"seed\"] == seed][\"selected_by\"].iloc[0],\n"
        "        include_linking=False,\n"
        "    )\n"
        "    records.append(row)\n"
        "    print(\n"
        "        f\"  chain break 평균 {row['qa_chain_break_fraction']:.4f}  \"\n"
        "        f\"feasible {row['feasible_fraction'] * 100:.2f}%  \"\n"
        "        f\"best feasible {row['best_feasible_objective']}\"\n"
        "    )\n"
        "\n"
        "results = FQ.to_schema_frame(records)\n"
        "print(f\"\\n{len(results)} 행, 컬럼 {len(results.columns)}개\")",
    ),
    (
        "md",
        "## 결과 저장\n\n"
        "**`qa_results.csv`(notebook 05)는 건드리지 않고 별도 파일 "
        "`qa_results_fixed_embedding.csv`에 저장합니다.** 컬럼 구성과 순서는 "
        "`qa_results.csv`와 동일하게 맞췄으므로, 나중에 두 파일을 그대로 "
        "concat 해서 비교할 수 있습니다.\n\n"
        "다만 스키마에 없는 컬럼 세 개를 **뒤에 덧붙였습니다.**\n\n"
        "| 컬럼 | 이유 |\n"
        "|---|---|\n"
        "| `seed` | 같은 instance를 여러 embedding으로 돌리므로, 없으면 행을 "
        "구분할 수 없습니다 |\n"
        "| `selected_by` | 그 seed가 어떤 지표로 뽑혔는지 |\n"
        "| `solver_id` | embedding이 어느 working graph 기준인지 |\n\n"
        "이 세 개가 불필요하면 `results.drop(columns=FQ.EXTRA_COLUMNS)`로 "
        "빼시면 됩니다. 다만 `seed`를 빼면 두 행이 물리 큐빗 수로만 구분되니 "
        "권하지 않습니다.\n\n"
        "alpha를 바꿔 다시 돌리면 같은 파일에 누적되며, "
        "`(instance, formulation, seed, qa_chain_strength_alpha)` 조합으로 "
        "구분됩니다.",
    ),
    (
        "code",
        "from src.persistence import save_table\n\n"
        "OUTPUT = PROCESSED_DIR / \"qa_results_fixed_embedding.csv\"\n"
        "if OUTPUT.exists():\n"
        "    previous = pd.read_csv(OUTPUT)\n"
        "    key = [\"instance\", \"formulation\", \"seed\", \"qa_chain_strength_alpha\"]\n"
        "    merged = pd.concat([previous, results], ignore_index=True)\n"
        "    merged = merged.drop_duplicates(subset=key, keep=\"last\")\n"
        "    merged = FQ.to_schema_frame(merged.to_dict(\"records\"))\n"
        "else:\n"
        "    merged = results\n"
        "save_table(merged, PROCESSED_DIR, \"qa_results_fixed_embedding.csv\")\n"
        "print(f\"저장: {OUTPUT} ({len(merged)} 행)\")\n"
        "print(\"컬럼:\", \", \".join(merged.columns))",
    ),
    (
        "md",
        "## embedding 품질과 QA 결과\n\n"
        "물리 큐빗이 적고 chain이 짧은 embedding이 실제로 더 나은 결과를 내는지 "
        "확인합니다.",
    ),
    (
        "code",
        "FQ.summarize(merged)",
    ),
    (
        "md",
        "## 4x4와 비교 — 이 notebook의 핵심\n\n"
        "chain 길이가 크게 다른 두 경우를 나란히 놓습니다.\n\n"
        "- **6x6도 chain break가 낮은데 feasible이 0** 이면, chain은 병목이 "
        "아니고 coefficient range가 지배적이라는 뜻입니다.\n"
        "- **6x6에서 chain break가 급증** 했다면 chain 길이도 유의한 요인입니다.\n\n"
        "4x4 결과는 notebook 05의 QA 결과에서 가져옵니다.",
    ),
    (
        "code",
        "# 주의: qa_results.csv는 chain 평균을 mean_chain_length로,\n"
        "#       embedding_study.csv는 avg_chain_length로 쓴다. 두 모듈을 따로\n"
        "#       만들면서 생긴 불일치인데, 지금 통일하면 이미 쌓인 CSV가 깨지므로\n"
        "#       여기서 각각의 이름을 맞춰 읽는다.\n"
        "try:\n"
        "    qa_prev = load_table(RAW_DIR, \"qa_results.csv\")\n"
        "    prev = qa_prev[\n"
        "        (qa_prev[\"instance\"] == \"4x4\")\n"
        "        & (qa_prev[\"formulation\"] == \"MS\")\n"
        "    ]\n"
        "    # linking 두 변형으로 돌렸다면 본 실험 설정(linking 제외)만 본다.\n"
        "    if \"linking\" in prev.columns:\n"
        "        prev = prev[~prev[\"linking\"].astype(bool)]\n"
        "    columns = [c for c in (\n"
        "        \"instance\", \"formulation\", \"linking\", \"status\",\n"
        "        \"embedding_status\", \"physical_qubits\", \"max_chain_length\",\n"
        "        \"mean_chain_length\", \"qa_chain_break_fraction\",\n"
        "        \"feasible_fraction\",\n"
        "    ) if c in prev.columns]\n"
        "    print(\"[notebook 05] MS 4x4 (linking 제외)\")\n"
        "    display(prev[columns])\n"
        "    if \"embedding_status\" in prev.columns:\n"
        "        status = set(prev[\"embedding_status\"].dropna().unique())\n"
        "        if status and status != {\"OK\"}:\n"
        "            print(\n"
        "                f\"경고: embedding_status={status}. QPU 실행 기록이 \"\n"
        "                f\"아니면 chain break 비교가 성립하지 않습니다. \"\n"
        "                f\"아래 셀로 4x4를 같은 경로에서 다시 돌리십시오.\"\n"
        "            )\n"
        "except FileNotFoundError:\n"
        "    print(\"notebook 05의 qa_results.csv가 없습니다.\")\n"
        "\n"
        "print(\"\\n[이번 실행] MS 6x6\")\n"
        "display(merged[[\n"
        "    \"seed\", \"physical_qubits\", \"max_chain_length\",\n"
        "    \"avg_chain_length\", \"qa_chain_break_fraction\",\n"
        "    \"feasible_fraction\", \"true_gap_percent\",\n"
        "]])",
    ),
    (
        "md",
        "## (선택) 4x4를 같은 경로로 다시 돌리기\n\n"
        "위 4x4 기록은 notebook 05에서 **다른 코드 경로**로 얻은 것입니다. "
        "`TARGET_INSTANCE`를 `\"4x4\"`로 바꾸고 이 notebook을 처음부터 다시 "
        "실행하면, embedding 품질만 다르고 **나머지 조건이 완전히 동일한 "
        "대조**가 됩니다.\n\n"
        "그러면 chain 길이가 chain break와 feasibility에 미치는 영향을 훨씬 "
        "깨끗하게 볼 수 있습니다. 4x4는 QPU 시간도 얼마 들지 않습니다.\n\n"
        "위 셀에서 `embedding_status`가 `OK`가 아니라는 경고가 떴다면 "
        "**반드시** 이 방법으로 다시 돌리십시오. 토큰 없이 기록된 행이면 "
        "chain break 값 자체가 없습니다.",
    ),
    (
        "md",
        "## 해석 메모\n\n"
        "아래는 결과를 보고 직접 채우십시오. 자동 판정은 두지 않았습니다. "
        "표본이 seed 2~3개뿐이라 통계적 결론을 내리기에 부족하고, 잘못된 자동 "
        "판정이 오히려 오해를 만들 수 있기 때문입니다.\n\n"
        "확인할 것\n\n"
        "1. chain break 평균이 4x4(0.025%)에 비해 얼마나 올랐는가\n"
        "2. feasible 비율이 0을 넘겼는가\n"
        "3. embedding 품질(큐빗 수, 최대 chain)과 chain break에 상관이 보이는가\n"
        "4. gap이 나왔다면 SA 결과(6x6 MS, 30.9%)와 비교해 어느 정도인가\n\n"
        "**주의**: alpha를 바꿔 다시 돌린 결과는 본 실험이 아니라 진단 실험으로 "
        "분류해 기록하십시오. 본 실험의 QA 설정은 alpha=1.5 고정입니다.",
    ),
]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(content) if kind == "md"
        else nbf.v4.new_code_cell(content)
        for kind, content in CELLS
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    path = NOTEBOOK_DIR / "30_MS_6x6_fixed_embedding_qa.ipynb"
    nbf.write(notebook, path)
    print(f"생성: {path.relative_to(PROJECT_ROOT)} ({len(CELLS)} 셀)")


if __name__ == "__main__":
    main()
