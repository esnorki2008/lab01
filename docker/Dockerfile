FROM apache/airflow:3.1.3-python3.11

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

# DVC en un entorno virtual aislado para no alterar las dependencias de Airflow
RUN python -m venv /opt/dvc-venv \
 && /opt/dvc-venv/bin/pip install --no-cache-dir "dvc==3.67.1" \
 && ln -s /opt/dvc-venv/bin/dvc /usr/local/bin/dvc

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
