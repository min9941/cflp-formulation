"""notebooks/31_MS_6x6_alpha_sweep.ipynb 를 생성한다.

실행:
    python scripts/build_alpha_notebook.py
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
pd.set_option("display.max_columns", 50)
print("설정 로드 완료")
'''

CELLS: list[tuple[str, str]] = [
    (
        "md",
        "# 31. MS 6x6 chain strength alpha sweep\n\n"
        "## 배경\n\n"
        "notebook 30에서 MS 6x6을 QPU에서 실행했습니다. 결과는 이랬습니다.\n\n"
        "| 항목 | 값 |\n"
        "|---|---|\n"
        "| embedding | 성공 (4,503 / 4,537 물리 큐빗) |\n"
        "| chain break | **1.3~2.1%** (낮음) |\n"
        "| feasible 비율 | **0%** (24회 전부) |\n"
        "| annealing time 20→200μs | 효과 없음 |\n"
        "| 같은 QUBO를 SA로 | **97.7% feasible** |\n\n"
        "chain break가 낮은데도 feasible이 0이라, **chain은 병목이 아니라는 것**이 "
        "확인됐습니다. auto_scale 후 objective 계수가 ICE 노이즈의 1/380 이하라서 "
        "QPU에 objective 정보가 전달되지 않는 것으로 보입니다.\n\n"
        "## 이 notebook이 시험하는 것\n\n"
        "chain break에 여유가 있다면 **chain strength를 낮춰 problem coupling에 "
        "dynamic range를 더 줄 수 있습니다.** alpha를 낮추면 chain이 약해져 "
        "chain break가 늘지만, 그 대가로 objective가 조금 더 보이게 됩니다.\n\n"
        "이 trade-off의 최적점이 있는지, 있다면 feasible 해가 나오는지 봅니다.",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        "## 먼저 알아둘 한계\n\n"
        "**alpha를 1 아래로 내려도 이득이 없습니다.**\n\n"
        "D-Wave의 auto_scale은 모든 계수를 `max(|chain coupling|, |problem "
        "coupling|)`로 나눕니다. `chain_strength = alpha × max|J|`이므로\n\n"
        "- `alpha >= 1` : chain이 최대값이라 분모 = `alpha × max|J|`\n"
        "- `alpha < 1` : problem coupling이 최대값이라 분모 = `max|J|` **고정**\n\n"
        "따라서 objective 계수의 정규화 크기는 alpha가 1 미만으로 내려가도 더 "
        "커지지 않습니다. **얻을 수 있는 최대 이득은 alpha=1.5 대비 1.5배**입니다.\n\n"
        "| alpha | objective 정규화 | ICE 노이즈 대비 |\n"
        "|---|---|---|\n"
        "| 0.5 | 2.25e-06 | 2.3e-04 |\n"
        "| **1.0** | **2.25e-06** | **2.3e-04** |\n"
        "| 1.5 (현재) | 1.50e-06 | 1.5e-04 |\n"
        "| 3.0 | 7.50e-07 | 7.5e-05 |\n\n"
        "필요한 개선은 **수백 배**인데 얻을 수 있는 건 1.5배입니다. 그래서 이 "
        "실험의 기대치는 낮습니다. 다만 alpha<1 구간을 넣는 이유는 **chain break가 "
        "언제 무너지는지**를 실측하기 위해서입니다. 그것 자체가 chain strength "
        "규칙의 타당성을 보여주는 자료가 됩니다.",
    ),
    (
        "md",
        "## 설정\n\n"
        "notebook 30에서 가장 좋았던 조건(`num_reads=1000`, `annealing_time=150`)을 "
        "고정하고 alpha만 훑습니다. embedding도 같은 두 seed를 씁니다.\n\n"
        "**이것은 진단 실험입니다.** 본 실험(notebook 05)의 QA 설정은 alpha=1.5 "
        "고정이며, 여기서 얻은 값을 본 실험 설정으로 채택하지 않습니다.",
    ),
    (
        "code",
        "TARGET_INSTANCE = \"6x6\"\n"
        "TARGET_FORMULATION = \"MS\"\n"
        "TARGET_TOPOLOGY = \"pegasus\"\n"
        "\n"
        "# notebook 30에서 확보한 embedding을 그대로 재사용한다.\n"
        "EMBED_TRIES = 10\n"
        "EMBED_TIMEOUT = 2400\n"
        "SEEDS = [2024, 2026]\n"
        "\n"
        "# notebook 30에서 가장 좋았던 조건으로 고정\n"
        "NUM_READS = 1000\n"
        "ANNEALING_TIME = 150.0\n"
        "\n"
        "# alpha grid. 1.0 아래는 이득이 없지만 chain break가 무너지는 지점을\n"
        "# 관측하기 위해 포함한다. 1.5는 본 실험 고정값이므로 반드시 넣는다.\n"
        "ALPHA_GRID = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]\n"
        "\n"
        "print(f\"대상          : {TARGET_INSTANCE} {TARGET_FORMULATION}\")\n"
        "print(f\"seed          : {SEEDS}\")\n"
        "print(f\"num_reads     : {NUM_READS}  (고정)\")\n"
        "print(f\"annealing_time: {ANNEALING_TIME}  (고정)\")\n"
        "print(f\"alpha grid    : {ALPHA_GRID}\")\n"
        "print(f\"총 실행       : {len(SEEDS) * len(ALPHA_GRID)}회\")",
    ),
    (
        "md",
        "## 예상 효과 미리 계산\n\n"
        "실행 전에 각 alpha에서 objective가 얼마나 보이게 되는지 계산합니다. "
        "실측 결과와 비교하면 모델이 맞는지 확인할 수 있습니다.",
    ),
    (
        "code",
        "from src.data_generator import CFLPInstance\n"
        "from src.qubo_builder import build_qubo, qubo_statistics\n"
        "from src.sa_solver import to_bqm\n\n"
        "instance = CFLPInstance.load(DATA_DIR / f\"{TARGET_INSTANCE}.json\")\n"
        "model = build_qubo(\n"
        "    instance,\n"
        "    TARGET_FORMULATION,\n"
        "    float(config[\"penalty\"][\"margin\"]),\n"
        "    int(config[\"encoding\"][\"precision\"]),\n"
        ")\n"
        "bqm = to_bqm(model)\n"
        "max_j = max(abs(v) for v in bqm.spin.quadratic.values())\n"
        "weighted = instance.transport_costs * instance.demands[:, None]\n"
        "objective_min = float(min(weighted.min(), instance.fixed_costs.min()))\n"
        "ICE_NOISE = 0.02\n"
        "\n"
        "rows = []\n"
        "for alpha in ALPHA_GRID:\n"
        "    denominator = max(alpha * max_j, max_j)   # auto_scale 분모\n"
        "    scaled = objective_min / denominator\n"
        "    rows.append({\n"
        "        \"alpha\": alpha,\n"
        "        \"chain_strength\": alpha * max_j,\n"
        "        \"auto_scale_denominator\": denominator,\n"
        "        \"objective_scaled\": scaled,\n"
        "        \"vs_ICE_noise\": scaled / ICE_NOISE,\n"
        "    })\n"
        "pd.DataFrame(rows)",
    ),
    (
        "md",
        "## embedding 확보\n\n"
        "notebook 30에서 저장한 mapping을 재사용합니다. 현재 working graph에서 "
        "유효한지 검사하고, 무효하면 자동으로 재탐색합니다.",
    ),
    (
        "code",
        "from src import fixed_embedding_qa as FQ\n\n"
        "embeddings = {}\n"
        "for seed in SEEDS:\n"
        "    print(f\"--- seed {seed} ---\")\n"
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
        "    print(\n"
        "        f\"  큐빗 {meta['physical_qubits']}, \"\n"
        "        f\"최대 chain {meta['max_chain_length']}\"\n"
        "    )",
    ),
    (
        "md",
        "## alpha sweep 실행\n\n"
        "embedding, QUBO, num_reads, annealing time이 모두 고정이고 **alpha만 "
        "바뀝니다.** 따라서 결과 차이는 전적으로 chain strength에서 옵니다.",
    ),
    (
        "code",
        "gurobi = pd.read_csv(RAW_DIR / \"gurobi_results.csv\")\n"
        "reference = float(\n"
        "    gurobi[(gurobi[\"instance\"] == TARGET_INSTANCE)\n"
        "           & (gurobi[\"gurobi_model\"] == TARGET_FORMULATION)][\"objective\"].iloc[0]\n"
        ")\n"
        "print(f\"Gurobi optimum {reference:.2f}\\n\")\n"
        "\n"
        "records = []\n"
        "for seed, (embedding, meta) in embeddings.items():\n"
        "    for alpha in ALPHA_GRID:\n"
        "        outcome = FQ.run_qa_with_embedding(\n"
        "            model=model,\n"
        "            instance=instance,\n"
        "            embedding=embedding,\n"
        "            config=config,\n"
        "            chain_strength_alpha=alpha,\n"
        "            solver_id=meta.get(\"solver_id\"),\n"
        "            num_reads=NUM_READS,\n"
        "            annealing_time=ANNEALING_TIME,\n"
        "        )\n"
        "        row = FQ.to_schema_row(\n"
        "            outcome=outcome,\n"
        "            meta=meta,\n"
        "            model=model,\n"
        "            instance=instance,\n"
        "            formulation=TARGET_FORMULATION,\n"
        "            selected_by=f\"alpha_sweep\",\n"
        "            include_linking=False,\n"
        "        )\n"
        "        records.append(row)\n"
        "        print(\n"
        "            f\"  seed {seed} alpha {alpha:4.2f}: \"\n"
        "            f\"chain break {row['qa_chain_break_fraction']:.4f}  \"\n"
        "            f\"feasible {row['feasible_fraction'] * 100:5.2f}%  \"\n"
        "            f\"최소 위반 {row['best_total_violation']}\"\n"
        "        )\n"
        "\n"
        "results = FQ.to_schema_frame(records)\n"
        "print(f\"\\n{len(results)}회 완료\")",
    ),
    (
        "md",
        "## 결과 저장\n\n"
        "**notebook 30과 별도 파일** `qa_results_alpha_sweep.csv`에 저장합니다. "
        "notebook 30은 본 실험 설정(alpha=1.5)의 QA 결과이고, 이 notebook은 "
        "alpha를 의도적으로 바꾸는 **진단 실험**이므로 섞이지 않게 분리합니다.\n\n"
        "| 파일 | 내용 |\n"
        "|---|---|\n"
        "| `qa_results.csv` | notebook 05, 본 실험 |\n"
        "| `qa_results_fixed_embedding.csv` | notebook 30, alpha=1.5 고정 |\n"
        "| `qa_results_alpha_sweep.csv` | **이 notebook, alpha 진단** |\n\n"
        "세 파일 모두 컬럼 구성이 같으므로 필요하면 그대로 concat 해서 비교할 수 "
        "있습니다. 중복 키는 "
        "`(instance, formulation, seed, alpha, annealing_time, num_reads)`이므로 "
        "조건을 바꿔 재실행하면 누적됩니다.",
    ),
    (
        "code",
        "from src.persistence import save_table\n\n"
        "FILENAME = \"qa_results_alpha_sweep.csv\"\n"
        "OUTPUT = PROCESSED_DIR / FILENAME\n"
        "KEY = [\"instance\", \"formulation\", \"seed\",\n"
        "       \"qa_chain_strength_alpha\", \"qa_annealing_time\", \"num_reads\"]\n"
        "if OUTPUT.exists():\n"
        "    previous = pd.read_csv(OUTPUT)\n"
        "    merged = pd.concat([previous, results], ignore_index=True)\n"
        "    merged = merged.drop_duplicates(subset=KEY, keep=\"last\")\n"
        "    merged = FQ.to_schema_frame(merged.to_dict(\"records\"))\n"
        "else:\n"
        "    merged = results\n"
        "save_table(merged, PROCESSED_DIR, FILENAME)\n"
        "print(f\"저장: {OUTPUT} ({len(merged)} 행)\")",
    ),
    (
        "md",
        "## alpha와 chain break / feasibility\n\n"
        "두 지표를 나란히 봅니다. **alpha를 낮추면 chain break가 오르는 것이 "
        "정상**이고, 그 대가로 feasible이 나오는지가 관건입니다.",
    ),
    (
        "code",
        "view = results[[\n"
        "    \"seed\", \"qa_chain_strength_alpha\", \"qa_chain_strength\",\n"
        "    \"qa_chain_break_fraction\", \"feasible_fraction\",\n"
        "    \"best_total_violation\", \"best_objective\", \"best_feasible_objective\",\n"
        "]].sort_values([\"seed\", \"qa_chain_strength_alpha\"])\n"
        "view",
    ),
    (
        "code",
        "pivot = results.pivot_table(\n"
        "    index=\"qa_chain_strength_alpha\",\n"
        "    columns=\"seed\",\n"
        "    values=[\"qa_chain_break_fraction\", \"feasible_fraction\",\n"
        "            \"best_total_violation\"],\n"
        ")\n"
        "pivot.round(4)",
    ),
    (
        "md",
        "## Figure\n\n"
        "왼쪽 축은 chain break, 오른쪽 축은 최소 위반량입니다. alpha가 낮아질 때 "
        "chain break가 오르는 지점과, 위반량이 줄어드는지를 함께 봅니다.",
    ),
    (
        "code",
        "import matplotlib\n"
        "matplotlib.use(\"Agg\")\n"
        "import matplotlib.pyplot as plt\n\n"
        "figure, axis = plt.subplots(figsize=(8.5, 4.8))\n"
        "twin = axis.twinx()\n"
        "colors = {SEEDS[0]: \"#2980b9\", SEEDS[-1]: \"#e67e22\"}\n"
        "for seed, group in results.groupby(\"seed\"):\n"
        "    group = group.sort_values(\"qa_chain_strength_alpha\")\n"
        "    color = colors.get(seed, \"#7f8c8d\")\n"
        "    axis.plot(group[\"qa_chain_strength_alpha\"],\n"
        "              group[\"qa_chain_break_fraction\"] * 100,\n"
        "              marker=\"o\", color=color, label=f\"seed {seed} chain break\")\n"
        "    twin.plot(group[\"qa_chain_strength_alpha\"],\n"
        "              group[\"best_total_violation\"],\n"
        "              marker=\"s\", linestyle=\"--\", color=color, alpha=0.6,\n"
        "              label=f\"seed {seed} violation\")\n"
        "axis.axvline(1.0, color=\"#c0392b\", linestyle=\":\", linewidth=1.2)\n"
        "axis.text(1.02, axis.get_ylim()[1] * 0.9, \"alpha=1\\n(이하는 이득 없음)\",\n"
        "          fontsize=8, color=\"#c0392b\")\n"
        "axis.axvline(1.5, color=\"#2c3e50\", linestyle=\":\", linewidth=1.2)\n"
        "axis.set_xlabel(\"chain strength alpha\")\n"
        "axis.set_ylabel(\"Chain break (%)\")\n"
        "twin.set_ylabel(\"Best total violation\")\n"
        "axis.set_title(\"Figure 20. Chain strength vs chain break and violation\")\n"
        "axis.grid(alpha=0.3)\n"
        "handles = axis.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]\n"
        "labels = axis.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]\n"
        "axis.legend(handles, labels, fontsize=8, loc=\"upper center\")\n"
        "figure.tight_layout()\n"
        "path = FIGURE_DIR / \"figure20_chain_strength_sweep.png\"\n"
        "figure.savefig(path, dpi=150, bbox_inches=\"tight\")\n"
        "plt.close(figure)\n"
        "from IPython.display import Image\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## 해석 메모\n\n"
        "직접 확인하십시오. 자동 판정은 두지 않았습니다.\n\n"
        "1. **feasible 해가 하나라도 나왔는가**\n"
        "   - 나왔다면 chain strength가 실제로 병목의 일부였다는 뜻입니다.\n"
        "   - 여전히 0이라면, 예상 계산대로 1.5배 개선으로는 부족하다는 것이 "
        "실측으로 확인된 것입니다.\n"
        "2. **alpha를 낮출 때 chain break가 언제 무너지는가**\n"
        "   - alpha=1.5에서 1.7%였습니다. 어느 지점에서 급증하는지 보십시오.\n"
        "   - 급증 지점이 1.0 근처면, 기존 규칙 `alpha=1.5`가 합리적인 여유를 "
        "둔 값이었다는 근거가 됩니다.\n"
        "3. **최소 위반량이 alpha에 반응하는가**\n"
        "   - notebook 30에서 24~87 범위였습니다. alpha를 낮춰 이 값이 내려가면 "
        "objective 가시성이 실제로 개선된 것입니다.\n"
        "4. **두 seed가 같은 경향을 보이는가**\n"
        "   - 경향이 다르면 embedding 구조에 따라 chain strength 최적점이 "
        "달라진다는 뜻이므로, 단일 규칙으로 고정하는 것의 한계를 시사합니다.\n\n"
        "**주의**: 여기서 얻은 alpha는 본 실험 설정으로 채택하지 마십시오. "
        "본 실험의 QA 설정은 `alpha=1.5` 고정이며, 이 notebook은 진단 실험입니다.",
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
    path = NOTEBOOK_DIR / "31_MS_6x6_alpha_sweep.ipynb"
    nbf.write(notebook, path)
    print(f"생성: {path.relative_to(PROJECT_ROOT)} ({len(CELLS)} 셀)")


if __name__ == "__main__":
    main()
