FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Utilisateur non-root pour la sécurité
RUN useradd -m scalper && chown -R scalper /app
USER scalper

CMD ["python", "scripts/prepare_features.py"] 