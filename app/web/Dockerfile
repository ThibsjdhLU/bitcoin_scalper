FROM python:3.11-slim

RUN useradd -m webuser
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install poetry
RUN if [ -f pyproject.toml ]; then poetry install --no-root; else pip install -r requirements.txt; fi

USER webuser

EXPOSE 8000 8501

CMD ["uvicorn", "app.web.api:app", "--host", "0.0.0.0", "--port", "8000"] 