"""notebooks/32_MS_6x6_violation_distribution.ipynb 를 생성한다.

실행:
    python scripts/build_violation_notebook.py
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

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
print("설정 로드 완료")
'''

CELLS: list[tuple[str, str]] = [
    (
        "md",
        "# 32. MS 6x6 제약 위반량 분포\n\n"
        "## 왜 이걸 보는가\n\n"
        "notebook 30, 31에서 QA는 40회 실행 전부 feasible 0%였습니다. 기록된 것은 "
        "**best sample 하나의 위반량**뿐입니다.\n\n"
        "| 실험 | 최소 위반량 |\n"
        "|---|---|\n"
        "| notebook 30 (alpha=1.5) | 24 ~ 87 |\n"
        "| notebook 31 (alpha 0.5~3.0) | 26 ~ 120 |\n\n"
        "그런데 \"최소 위반 26\"이라는 숫자 하나로는 두 상황을 구분할 수 "
        "없습니다.\n\n"
        "| 상황 | 의미 |\n"
        "|---|---|\n"
        "| 낮은 위반이 **두텁게** 쌓여 있는데 0만 못 찍음 | 거의 다 왔다. 정밀도를 "
        "조금만 올려도 feasible이 나올 수 있다 |\n"
        "| 26이 **외톨이**이고 대부분 훨씬 큼 | 구조적으로 멀다. 정밀도 개선으로 "
        "넘을 수 있는 거리가 아니다 |\n\n"
        "등식 제약이라 위반 0만이 feasible이므로, **0 근처의 밀도**가 핵심입니다.\n\n"
        "## 비교 대상\n\n"
        "같은 QUBO를 SA로 풀면 97.7%가 feasible입니다. SA와 QA의 위반량 분포를 "
        "나란히 놓으면 하드웨어가 무엇을 잃는지 한눈에 보입니다.",
    ),
    ("code", BOOTSTRAP),
    (
        "md",
        "## 설정\n\n"
        "`ALPHA_LIST`에 넣은 각 alpha로 QA를 다시 실행해 **전체 sample의 위반량**을 "
        "수집합니다. notebook 30/31은 best만 저장했으므로 재실행이 필요합니다.\n\n"
        "`RUN_QA=False`로 두면 QA를 건너뛰고 SA 분포만 봅니다. QPU 없이 먼저 "
        "확인하실 때 쓰십시오.",
    ),
    (
        "code",
        "TARGET_INSTANCE = \"6x6\"\n"
        "TARGET_FORMULATION = \"MS\"\n"
        "TARGET_TOPOLOGY = \"pegasus\"\n"
        "\n"
        "RUN_QA = True\n"
        "SEEDS = [2024]                  # embedding seed\n"
        "ALPHA_LIST = [0.7, 1.5]         # 31에서 최선(0.7)과 본 실험값(1.5)\n"
        "NUM_READS = 1000\n"
        "ANNEALING_TIME = 150.0\n"
        "EMBED_TRIES = 10\n"
        "EMBED_TIMEOUT = 2400\n"
        "\n"
        "print(f\"QA 실행 : {RUN_QA}\")\n"
        "print(f\"alpha   : {ALPHA_LIST}\")\n"
        "print(f\"seed    : {SEEDS}\")",
    ),
    (
        "md",
        "## QUBO 준비",
    ),
    (
        "code",
        "from src.data_generator import CFLPInstance\n"
        "from src.qubo_builder import build_qubo\n"
        "from src import fixed_embedding_qa as FQ\n\n"
        "instance = CFLPInstance.load(DATA_DIR / f\"{TARGET_INSTANCE}.json\")\n"
        "model = build_qubo(\n"
        "    instance,\n"
        "    TARGET_FORMULATION,\n"
        "    float(config[\"penalty\"][\"margin\"]),\n"
        "    int(config[\"encoding\"][\"precision\"]),\n"
        ")\n"
        "tolerance = float(config[\"feasibility\"][\"tolerance\"])\n"
        "print(f\"변수 {model.num_variables}, 이차항 {model.num_quadratic_terms}\")\n"
        "print(f\"총 수요 {instance.total_demand}  (위반량을 이 값과 비교하면 "
        "상대적 크기를 알 수 있습니다)\")",
    ),
    (
        "md",
        "## SA 기준선\n\n"
        "본 실험과 동일한 SA 설정으로 위반량 분포를 구합니다. 이것이 "
        "**\"QUBO 자체는 풀 수 있다\"**는 기준선입니다.",
    ),
    (
        "code",
        "import neal\n"
        "from src.sa_solver import to_bqm\n\n"
        "bqm = to_bqm(model)\n"
        "sa_sampleset = neal.SimulatedAnnealingSampler().sample(\n"
        "    bqm,\n"
        "    num_reads=int(config[\"sa\"][\"num_reads\"]),\n"
        "    num_sweeps=int(config[\"sa\"][\"num_sweeps\"]),\n"
        "    seed=int(config[\"sa\"][\"seed\"]),\n"
        ")\n"
        "sa_dist = FQ.violation_distribution(model, instance, sa_sampleset, tolerance)\n"
        "sa_stat = FQ.violation_summary(model, instance, sa_sampleset, tolerance)\n"
        "print(\"[SA] 위반량 분포\")\n"
        "display(sa_dist)\n"
        "print(\"[SA] 요약\", {k: round(v, 2) if isinstance(v, float) else v\n"
        "                     for k, v in sa_stat.items()})",
    ),
    (
        "md",
        "## QA 실행 및 위반량 수집\n\n"
        "embedding은 notebook 30에서 저장한 것을 재사용합니다. alpha만 바꿔가며 "
        "전체 sample의 위반량을 수집합니다.",
    ),
    (
        "code",
        "import time\n\n"
        "distributions = {\"SA\": sa_dist.set_index(\"range\")[\"count\"]}\n"
        "summaries = [dict(source=\"SA\", **sa_stat)]\n"
        "\n"
        "def log(message):\n"
        "    \"\"\"어디서 시간이 걸리는지 보이도록 진행 상황을 찍는다.\"\"\"\n"
        "    print(f\"[{time.strftime('%H:%M:%S')}] {message}\", flush=True)\n"
        "\n"
        "if RUN_QA:\n"
        "    log(\"dwave.system import 중\")\n"
        "    from dwave.system import DWaveSampler, FixedEmbeddingComposite\n"
        "    from src import qa_solver\n"
        "    log(\"import 완료\")\n"
        "\n"
        "    for seed in SEEDS:\n"
        "        log(f\"seed {seed}: embedding 확보 시작\")\n"
        "        embedding, meta = FQ.obtain_embedding(\n"
        "            instance=instance,\n"
        "            formulation=TARGET_FORMULATION,\n"
        "            config=config,\n"
        "            seed=seed,\n"
        "            tries=EMBED_TRIES,\n"
        "            timeout=EMBED_TIMEOUT,\n"
        "            topology=TARGET_TOPOLOGY,\n"
        "            embedding_dir=EMBEDDING_DIR,\n"
        "        )\n"
        "        log(\n"
        "            f\"seed {seed}: embedding 확보 완료 \"\n"
        "            f\"(큐빗 {meta['physical_qubits']}, \"\n"
        "            f\"재사용={meta.get('reused')})\"\n"
        "        )\n"
        "\n"
        "        log(f\"solver 접속: {meta['solver_id']}\")\n"
        "        sampler = DWaveSampler(solver=meta[\"solver_id\"])\n"
        "        composite = FixedEmbeddingComposite(sampler, embedding)\n"
        "        log(\"composite 준비 완료\")\n"
        "\n"
        "        for alpha in ALPHA_LIST:\n"
        "            chain_strength = qa_solver.compute_chain_strength(bqm, alpha)\n"
        "            log(\n"
        "                f\"alpha={alpha}: QPU 제출 \"\n"
        "                f\"(chain_strength={chain_strength:.3e}) — \"\n"
        "                f\"큐 대기가 길 수 있습니다\"\n"
        "            )\n"
        "            started = time.perf_counter()\n"
        "            sampleset = composite.sample(\n"
        "                bqm,\n"
        "                num_reads=NUM_READS,\n"
        "                annealing_time=ANNEALING_TIME,\n"
        "                chain_strength=chain_strength,\n"
        "                answer_mode=\"raw\",\n"
        "            )\n"
        "            sampleset.resolve()\n"
        "            log(f\"alpha={alpha}: QPU 응답 수신 ({time.perf_counter()-started:.1f}s)\")\n"
        "\n"
        "            label = f\"QA a={alpha}\"\n"
        "            started = time.perf_counter()\n"
        "            violations, weights = FQ.sample_violations(\n"
        "                model, instance, sampleset, tolerance\n"
        "            )\n"
        "            dist = FQ.violation_distribution(\n"
        "                model, instance, sampleset, tolerance\n"
        "            )\n"
        "            stat = FQ.violation_summary(\n"
        "                model, instance, sampleset, tolerance\n"
        "            )\n"
        "            log(f\"alpha={alpha}: 위반량 분석 완료 ({time.perf_counter()-started:.1f}s)\")\n"
        "\n"
        "            distributions[label] = dist.set_index(\"range\")[\"count\"]\n"
        "            summaries.append(dict(source=label, seed=seed, **stat))\n"
        "            print(\n"
        "                f\"  -> 최소 위반 {stat['min_violation']:.0f}, \"\n"
        "                f\"위반<=3 인 sample {stat['num_violation_le_3']}개, \"\n"
        "                f\"feasible {int(dist.iloc[0]['count'])}개\"\n"
        "            )\n"
        "    log(\"전체 완료\")\n"
        "else:\n"
        "    print(\"RUN_QA=False 이므로 SA 분포만 봅니다.\")",
    ),
    (
        "md",
        "## 분포 비교\n\n"
        "각 구간의 sample 수입니다. **0 근처 구간에 얼마나 쌓여 있는지**가 "
        "핵심입니다.",
    ),
    (
        "code",
        "comparison = pd.DataFrame(distributions).fillna(0).astype(int)\n"
        "comparison",
    ),
    (
        "code",
        "summary_frame = pd.DataFrame(summaries)\n"
        "summary_frame.round(2)",
    ),
    (
        "md",
        "## Figure\n\n"
        "가로축이 위반량 구간, 세로축이 sample 수(log)입니다. SA는 0에 몰려 "
        "있고 QA가 어디에 몰려 있는지 봅니다.",
    ),
    (
        "code",
        "import matplotlib\n"
        "matplotlib.use(\"Agg\")\n"
        "import matplotlib.pyplot as plt\n\n"
        "figure, axis = plt.subplots(figsize=(9.5, 4.8))\n"
        "labels = list(comparison.index)\n"
        "positions = np.arange(len(labels), dtype=float)\n"
        "width = 0.8 / max(len(comparison.columns), 1)\n"
        "for offset, column in enumerate(comparison.columns):\n"
        "    axis.bar(\n"
        "        positions - 0.4 + width * (offset + 0.5),\n"
        "        comparison[column].clip(lower=0.1),\n"
        "        width=width, label=column,\n"
        "    )\n"
        "axis.set_yscale(\"log\")\n"
        "axis.set_xticks(positions)\n"
        "axis.set_xticklabels(labels, rotation=30, ha=\"right\")\n"
        "axis.set_xlabel(\"Total constraint violation\")\n"
        "axis.set_ylabel(\"Number of samples (log)\")\n"
        "axis.set_title(\"Figure 21. Constraint violation distribution: SA vs QA\")\n"
        "axis.grid(axis=\"y\", alpha=0.3)\n"
        "axis.legend(fontsize=8)\n"
        "figure.tight_layout()\n"
        "path = FIGURE_DIR / \"figure21_violation_distribution.png\"\n"
        "figure.savefig(path, dpi=150, bbox_inches=\"tight\")\n"
        "plt.close(figure)\n"
        "from IPython.display import Image\n"
        "Image(filename=str(path))",
    ),
    (
        "md",
        "## 저장",
    ),
    (
        "code",
        "from src.persistence import save_table\n\n"
        "save_table(\n"
        "    comparison.reset_index(), PROCESSED_DIR,\n"
        "    \"ms6x6_violation_distribution.csv\",\n"
        ")\n"
        "save_table(\n"
        "    summary_frame, PROCESSED_DIR, \"ms6x6_violation_summary.csv\"\n"
        ")\n"
        "print(\"저장 완료\")",
    ),
    (
        "md",
        "## 해석 메모\n\n"
        "직접 확인하십시오.\n\n"
        "1. **QA의 위반량 최빈 구간이 어디인가**\n"
        "   - 0~5 구간에 두텁게 쌓여 있다면 \"거의 다 왔다\"는 뜻입니다. 정밀도를 "
        "조금만 개선해도 feasible이 나올 수 있습니다.\n"
        "   - 20~50 이상에 몰려 있다면 best sample의 낮은 위반이 외톨이였다는 "
        "뜻이고, 구조적으로 멉니다.\n"
        "2. **`num_violation_le_3`가 몇 개인가**\n"
        "   - 1,000개 중 한 자릿수면 우연에 가깝습니다.\n"
        "   - 수십 개라면 분포가 0 쪽으로 밀려 있다는 뜻입니다.\n"
        "3. **alpha 0.7과 1.5의 분포 차이**\n"
        "   - notebook 31에서 최소 위반이 40% 줄었는데, 분포 전체가 이동한 것인지 "
        "꼬리만 좋아진 것인지 확인하십시오.\n"
        "4. **SA와의 격차**\n"
        "   - SA는 97.7%가 위반 0입니다. QA 분포가 그것과 얼마나 떨어져 있는지가 "
        "하드웨어가 잃는 양입니다.\n\n"
        "**주의**: 이 notebook은 진단 실험입니다. 여기서 얻은 alpha나 설정을 본 "
        "실험 값으로 채택하지 마십시오.",
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
    path = NOTEBOOK_DIR / "32_MS_6x6_violation_distribution.ipynb"
    nbf.write(notebook, path)
    print(f"생성: {path.relative_to(PROJECT_ROOT)} ({len(CELLS)} 셀)")


if __name__ == "__main__":
    main()
