from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"
WORK = f"{PROJECT}/airflow/work"
PY = "/opt/airflow/venv/bin/python"

with DAG(
    dag_id="recsys_lsh_birank",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["capstone"],
) as dag:
    data_preprocessing = BashOperator(
        task_id="data_preprocessing",
        bash_command=f"""set -e
mkdir -p {WORK}/data/raw {WORK}/data/processed
rm -rf {WORK}/src && cp -r {PROJECT}/src {WORK}/src
cd {WORK}
if [ ! -d data/raw/ml-latest-small ]; then
  curl -L -o data/raw/ml.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
  {PY} -m zipfile -e data/raw/ml.zip data/raw
fi
{PY} src/data_preprocessing.py""",
    )
    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {WORK} && {PY} src/build_features.py",
    )
    lsh_candidates = BashOperator(
        task_id="lsh_candidates",
        bash_command=f"cd {WORK} && {PY} src/lsh_candidates.py",
    )
    finalize_candidates = BashOperator(
        task_id="finalize_candidates",
        bash_command=f"cd {WORK} && {PY} src/finalize_candidates.py",
    )
    birank_rerank = BashOperator(
        task_id="birank_rerank",
        bash_command=f"cd {WORK} && {PY} src/birank_rerank.py rerank --data data/processed --weight rating --alpha 0.7 --beta 0.1 --topn 20",
    )
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"cd {WORK} && {PY} src/evaluate.py --data data/processed --out {WORK}/results",
    )

    data_preprocessing >> build_features >> lsh_candidates >> finalize_candidates >> birank_rerank >> evaluate
