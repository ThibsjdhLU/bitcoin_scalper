FROM python:3.11-slim

# Sécurité : utilisateur non-root
RUN useradd -m botuser
WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copie du code
COPY . .

# Installation Poetry (préparation migration)
RUN pip install poetry

# Installation des dépendances
RUN if [ -f pyproject.toml ]; then poetry install --no-root; else pip install -r requirements.txt; fi

USER botuser

CMD ["python", "app/main.py"] 