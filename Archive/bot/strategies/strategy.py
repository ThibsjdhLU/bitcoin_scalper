#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stratégie de trading utilisant les indicateurs techniques
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .indicators import TechnicalIndicators


class BaseStrategy:
    """Classe pour gérer la stratégie de trading"""

    def __init__(self, config: Dict):
        """
        Initialise la stratégie de trading

        Args:
            config: Configuration du bot
        """
        self.config = config
        self.indicators = TechnicalIndicators()
        self.prices: List[float] = []
        self.max_prices = 100  # Nombre maximum de prix à conserver

    def update_prices(self, price: float) -> None:
        """
        Met à jour la liste des prix

        Args:
            price: Nouveau prix
        """
        self.prices.append(price)
        if len(self.prices) > self.max_prices:
            self.prices.pop(0)

    def analyze_market(self) -> Dict:
        """
        Analyse le marché et génère un signal de trading.

        Returns:
            Dict: Signal de trading et indicateurs
        """
        try:
            # Récupération des données
            data = self.exchange.get_historical_data("BTCUSD", "1m", 100)
            if data.empty:
                print("Erreur: Pas de données disponibles pour l'analyse")
                return {"signal": "NONE", "strength": 0, "indicators": {}}

            # Calcul des indicateurs
            rsi = self.indicators.calculate_rsi(data["close"])
            macd, signal, hist = self.indicators.calculate_macd(data["close"])
            upper_band, sma, lower_band = self.indicators.calculate_bollinger_bands(
                data["close"]
            )
            support, resistance = self.indicators.calculate_support_resistance(data)

            # Génération des signaux
            signals = []
            strength = 0

            print("\n=== Analyse des indicateurs ===")

            # Signal RSI
            if rsi < 35:  # Assoupli de 30 à 35
                signals.append("BUY")
                strength += 1
                print(f"Signal RSI: SURVENTE (RSI = {rsi:.2f}) -> BUY")
            elif rsi > 65:  # Assoupli de 70 à 65
                signals.append("SELL")
                strength -= 1
                print(f"Signal RSI: SURACHAT (RSI = {rsi:.2f}) -> SELL")
            else:
                print(f"RSI neutre: {rsi:.2f}")

            # Signal MACD
            if macd > signal:
                signals.append("BUY")
                strength += 1
                print(
                    f"Signal MACD: CROISEMENT HAUSSIER (MACD = {macd:.2f}, Signal = {signal:.2f}) -> BUY"
                )
            elif macd < signal:
                signals.append("SELL")
                strength -= 1
                print(
                    f"Signal MACD: CROISEMENT BAISSIER (MACD = {macd:.2f}, Signal = {signal:.2f}) -> SELL"
                )
            else:
                print(f"MACD neutre: {macd:.2f}")

            # Signal Bandes de Bollinger
            current_price = data["close"].iloc[-1]
            if current_price < lower_band * 1.01:  # Ajout d'une marge de 1%
                signals.append("BUY")
                strength += 1
                print(
                    f"Signal BB: PRIX PROCHE DE LA BANDE INFÉRIEURE ({current_price:.2f} < {lower_band * 1.01:.2f}) -> BUY"
                )
            elif current_price > upper_band * 0.99:  # Ajout d'une marge de 1%
                signals.append("SELL")
                strength -= 1
                print(
                    f"Signal BB: PRIX PROCHE DE LA BANDE SUPÉRIEURE ({current_price:.2f} > {upper_band * 0.99:.2f}) -> SELL"
                )
            else:
                print(
                    f"BB neutre: Prix = {current_price:.2f}, Bande inf = {lower_band:.2f}, Bande sup = {upper_band:.2f}"
                )

            # Signal Support/Résistance
            if current_price < support * 1.01:  # Ajout d'une marge de 1%
                signals.append("BUY")
                strength += 1
                print(
                    f"Signal S/R: PRIX PROCHE DU SUPPORT ({current_price:.2f} < {support * 1.01:.2f}) -> BUY"
                )
            elif current_price > resistance * 0.99:  # Ajout d'une marge de 1%
                signals.append("SELL")
                strength -= 1
                print(
                    f"Signal S/R: PRIX PROCHE DE LA RÉSISTANCE ({current_price:.2f} > {resistance * 0.99:.2f}) -> SELL"
                )
            else:
                print(
                    f"S/R neutre: Prix = {current_price:.2f}, Support = {support:.2f}, Résistance = {resistance:.2f}"
                )

            # Décision finale
            final_signal = "NONE"
            if signals:
                buy_count = signals.count("BUY")
                sell_count = signals.count("SELL")

                print(f"\nRésumé des signaux:")
                print(f"Signaux d'achat: {buy_count}")
                print(f"Signaux de vente: {sell_count}")
                print(f"Force du signal: {strength}")

                if buy_count > sell_count and strength >= 1:  # Assoupli de 2 à 1
                    final_signal = "BUY"
                    print("DÉCISION FINALE: BUY (Majorité d'achats et force >= 1)")
                elif sell_count > buy_count and strength <= -1:  # Assoupli de -2 à -1
                    final_signal = "SELL"
                    print("DÉCISION FINALE: SELL (Majorité de ventes et force <= -1)")
                else:
                    print("DÉCISION FINALE: NONE (Pas assez de force dans le signal)")
            else:
                print("Aucun signal généré")

            return {
                "signal": final_signal,
                "strength": strength,
                "indicators": {
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": signal,
                    "macd_hist": hist,
                    "bb_upper": upper_band,
                    "bb_middle": sma,
                    "bb_lower": lower_band,
                    "support": support,
                    "resistance": resistance,
                },
            }

        except Exception as e:
            print(f"Erreur lors de l'analyse du marché: {str(e)}")
            return {"signal": "NONE", "strength": 0, "indicators": {}}

    def should_trade(self, analysis: Dict) -> bool:
        """
        Détermine si un trade devrait être exécuté

        Args:
            analysis: Résultats de l'analyse du marché

        Returns:
            True si un trade devrait être exécuté
        """
        if analysis["signal"] == "WAIT":
            return False

        # Vérification de la force du signal
        min_strength = self.config.get("min_signal_strength", 2)
        if abs(analysis["strength"]) < min_strength:
            return False

        # Vérification des conditions supplémentaires
        indicators = analysis["indicators"]

        # Vérification RSI
        rsi = indicators["rsi"]
        if analysis["signal"] == "BUY" and rsi > 70:
            return False
        if analysis["signal"] == "SELL" and rsi < 30:
            return False

        # Vérification MACD
        if analysis["signal"] == "BUY" and indicators["macd_hist"] < 0:
            return False
        if analysis["signal"] == "SELL" and indicators["macd_hist"] > 0:
            return False

        return True
