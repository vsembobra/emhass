FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pandas pvlib lightgbm scikit-learn certifi

WORKDIR /opt
COPY src/ /opt/emhass_src/src/
ENV PYTHONPATH=/opt/emhass_src/src

COPY models/ /opt/models_default/
COPY app/ /opt/app/
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]
