FROM python:3.9-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    libfreetype6 \
    libx11-6 \
    libxext-dev \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installation de MetaTrader 5
RUN wget -qO- https://www.metatrader5.com/en/terminal/help/start_advanced_install_deb | bash

# Configuration de l'environnement Python
WORKDIR /app

# Copie des requirements d'abord pour optimiser le cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Création des répertoires nécessaires
RUN mkdir -p data logs src/dashboard

# Copie du code source
COPY . .

# Installation du package en mode développement
RUN pip install -e .

# Vérification des permissions
RUN chmod -R 755 /app

# Commande par défaut
CMD ["python", "run_bot.py"] 