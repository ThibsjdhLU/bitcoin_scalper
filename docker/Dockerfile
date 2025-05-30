FROM python:3.11-slim

# Sécurité : utilisateur non-root
RUN useradd -m botuser
WORKDIR /app

# Dépendances système minimales
RUN rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y --no-install-recommends build-essential gnupg lsb-release curl && rm -rf /var/lib/apt/lists/*

# Installation de TimescaleDB (pour l'intégration directe si nécessaire)
RUN echo "deb https://packagecloud.io/timescale/timescaledb/debian/ $(lsb_release -c -s) main" | tee /etc/apt/sources.list.d/timescaledb.list && \
    curl -SsL https://packagecloud.io/timescale/timescaledb/gpgkey | apt-key add - && \
    apt-get update && apt-get install -y timescaledb-2-postgresql-$(dpkg -s postgresql | grep '^Version:' | cut -d'.' -f1,2 | cut -d':' -f2) && \
    rm -rf /var/lib/apt/lists/*

# Configuration de PostgreSQL pour charger l'extension TimescaleDB (dans le conteneur)
RUN if [ -f /etc/postgresql/*/main/postgresql.conf ]; then \
        find /etc/postgresql -name postgresql.conf -exec sed -i "s/^#*\s*shared_preload_libraries = .*$/shared_preload_libraries = 'timescaledb'/g" {} \; \
    fi

# Copie du code
COPY . .

# Installation Poetry (préparation migration)
RUN pip install poetry

# Installation des dépendances
RUN if [ -f pyproject.toml ]; then poetry install --no-root; else pip install -r requirements.txt; fi

USER botuser

CMD ["python", "app/main.py"] 