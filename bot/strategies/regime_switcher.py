"""
Adaptation de la stratégie en fonction du régime de marché détecté
"""

from datetime import datetime

import numpy as np
import pandas as pd


class RegimeSwitcher:
    """
    Classe pour adapter la stratégie en fonction du régime de marché
    """

    def __init__(self, regime_detector, strategy):
        """
        Initialise l'adaptateur de régime

        Args:
            regime_detector: Instance du détecteur de régime (HMM ou GMM)
            strategy: Instance de la stratégie à adapter
        """
        self.regime_detector = regime_detector
        self.strategy = strategy
        self.current_regime = None
        self.regime_history = []

        # Paramètres par défaut pour chaque régime
        self.regime_parameters = {
            "bear": {
                "rsi_overbought": 60,  # Plus conservateur en marché baissier
                "rsi_oversold": 20,
                "stop_loss_pips": 100,
                "take_profit_pips": 50,
                "volume_threshold": 2.0,
                "atr_multiplier": 2.0,
            },
            "sideways": {
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "stop_loss_pips": 75,
                "take_profit_pips": 75,
                "volume_threshold": 1.5,
                "atr_multiplier": 1.5,
            },
            "bull": {
                "rsi_overbought": 80,  # Plus agressif en marché haussier
                "rsi_oversold": 40,
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
                "volume_threshold": 1.2,
                "atr_multiplier": 1.2,
            },
        }

    def update_regime(self, data):
        """
        Met à jour le régime détecté

        Args:
            data (pd.DataFrame): Données de prix récentes

        Returns:
            str: Régime détecté
        """
        # Prédire le régime pour les données récentes
        regimes = self.regime_detector.predict(data)

        # Utiliser le dernier régime prédit
        current_regime_idx = regimes[-1]
        current_regime = self.regime_detector.regime_names.get(
            current_regime_idx, f"regime_{current_regime_idx}"
        )

        # Enregistrer le régime actuel
        self.current_regime = current_regime

        # Ajouter à l'historique
        self.regime_history.append(
            {"timestamp": datetime.now(), "regime": current_regime}
        )

        return current_regime

    def adapt_strategy(self, data):
        """
        Adapte la stratégie au régime détecté

        Args:
            data (pd.DataFrame): Données de prix récentes

        Returns:
            dict: Paramètres adaptés
        """
        # Mettre à jour le régime
        regime = self.update_regime(data)

        # Récupérer les paramètres pour ce régime
        if regime in self.regime_parameters:
            parameters = self.regime_parameters[regime]
        else:
            # Utiliser les paramètres par défaut si le régime n'est pas reconnu
            parameters = self.regime_parameters["sideways"]

        # Adapter la stratégie
        self._apply_parameters(parameters)

        return parameters

    def _apply_parameters(self, parameters):
        """
        Applique les paramètres à la stratégie

        Args:
            parameters (dict): Paramètres à appliquer
        """
        # Mettre à jour les paramètres de la stratégie
        for key, value in parameters.items():
            if hasattr(self.strategy, key):
                setattr(self.strategy, key, value)

    def get_regime_statistics(self):
        """
        Calcule les statistiques des régimes

        Returns:
            dict: Statistiques des régimes
        """
        if not self.regime_history:
            return {}

        # Convertir l'historique en DataFrame
        history_df = pd.DataFrame(self.regime_history)

        # Calculer les statistiques
        regime_counts = history_df["regime"].value_counts()
        regime_percentages = regime_counts / len(history_df) * 100

        # Calculer la durée moyenne de chaque régime
        regime_durations = {}

        for regime in regime_counts.index:
            regime_data = history_df[history_df["regime"] == regime]

            if len(regime_data) > 1:
                # Calculer la durée moyenne entre les changements de régime
                timestamps = regime_data["timestamp"].values
                durations = np.diff(timestamps)
                avg_duration = np.mean(durations)
                regime_durations[regime] = avg_duration
            else:
                regime_durations[regime] = 0

        return {
            "counts": regime_counts.to_dict(),
            "percentages": regime_percentages.to_dict(),
            "durations": regime_durations,
        }
