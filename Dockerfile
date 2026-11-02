FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock poetry.toml ./
RUN pip install --no-cache-dir poetry==2.2.1 \
    && poetry sync --only main --no-interaction --no-ansi

COPY src/ src/
COPY data/telco_customer_churn.csv data/telco_customer_churn.csv

ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--automl-trials", "20"]
