"""
Tests de perturbation pour évaluer la robustesse de la stratégie
"""

from datetime import datetime

import numpy as np
import pandas as pd

from .market_generator import MarketGenerator


class PerturbationTest:
    """
    Classe pour tester la robustesse de la stratégie face à des perturbations
    """

    def __init__(self, strategy, base_data):
        """
        Initialise le test de perturbation

        Args:
            strategy: Instance de la stratégie à tester
            base_data (pd.DataFrame): Données de base pour les tests
        """
        self.strategy = strategy
        self.base_data = base_data
        self.market_generator = MarketGenerator(base_data)
        self.results = {}

    def run_slippage_test(
        self, slippage_percentages=[0.5, 1.0, 2.0], duration_minutes=30
    ):
        """
        Teste la stratégie face à différents niveaux de slippage

        Args:
            slippage_percentages (list): Liste des pourcentages de slippage à tester
            duration_minutes (int): Durée du slippage en minutes

        Returns:
            dict: Résultats des tests
        """
        results = {}

        for slippage in slippage_percentages:
            # Générer les données avec slippage
            perturbed_data = self.market_generator.generate_slippage_scenario(
                slippage_percent=slippage, duration_minutes=duration_minutes
            )

            # Exécuter la stratégie sur les données perturbées
            performance = self._run_strategy(perturbed_data)

            # Enregistrer les résultats
            results[f"slippage_{slippage}%"] = performance

        self.results["slippage"] = results
        return results

    def run_gap_test(self, gap_percentages=[1.0, 2.0, 5.0], directions=["up", "down"]):
        """
        Teste la stratégie face à différents gaps de prix

        Args:
            gap_percentages (list): Liste des pourcentages de gap à tester
            directions (list): Directions des gaps à tester

        Returns:
            dict: Résultats des tests
        """
        results = {}

        for gap in gap_percentages:
            for direction in directions:
                # Générer les données avec gap
                perturbed_data = self.market_generator.generate_gap_scenario(
                    gap_percent=gap, gap_direction=direction
                )

                # Exécuter la stratégie sur les données perturbées
                performance = self._run_strategy(perturbed_data)

                # Enregistrer les résultats
                results[f"gap_{gap}%_{direction}"] = performance

        self.results["gap"] = results
        return results

    def run_volatility_test(self, multipliers=[2.0, 3.0, 5.0], duration_minutes=60):
        """
        Teste la stratégie face à des pics de volatilité

        Args:
            multipliers (list): Liste des multiplicateurs de volatilité à tester
            duration_minutes (int): Durée du pic en minutes

        Returns:
            dict: Résultats des tests
        """
        results = {}

        for multiplier in multipliers:
            # Générer les données avec pic de volatilité
            perturbed_data = self.market_generator.generate_volatility_spike(
                volatility_multiplier=multiplier, duration_minutes=duration_minutes
            )

            # Exécuter la stratégie sur les données perturbées
            performance = self._run_strategy(perturbed_data)

            # Enregistrer les résultats
            results[f"volatility_{multiplier}x"] = performance

        self.results["volatility"] = results
        return results

    def run_liquidity_test(
        self, volume_reductions=[0.5, 0.7, 0.9], duration_minutes=120
    ):
        """
        Teste la stratégie face à des crises de liquidité

        Args:
            volume_reductions (list): Liste des réductions de volume à tester
            duration_minutes (int): Durée de la crise en minutes

        Returns:
            dict: Résultats des tests
        """
        results = {}

        for reduction in volume_reductions:
            # Générer les données avec crise de liquidité
            perturbed_data = self.market_generator.generate_liquidity_crisis(
                volume_reduction=reduction, duration_minutes=duration_minutes
            )

            # Exécuter la stratégie sur les données perturbées
            performance = self._run_strategy(perturbed_data)

            # Enregistrer les résultats
            results[f"liquidity_{int(reduction*100)}%"] = performance

        self.results["liquidity"] = results
        return results

    def run_all_tests(self):
        """
        Exécute tous les tests de perturbation

        Returns:
            dict: Résultats de tous les tests
        """
        self.run_slippage_test()
        self.run_gap_test()
        self.run_volatility_test()
        self.run_liquidity_test()

        return self.results

    def _run_strategy(self, data):
        """
        Exécute la stratégie sur les données données

        Args:
            data (pd.DataFrame): Données sur lesquelles exécuter la stratégie

        Returns:
            dict: Performance de la stratégie
        """
        # Sauvegarder les données originales
        original_data = self.strategy.data

        # Remplacer les données par les données perturbées
        self.strategy.data = data

        # Exécuter la stratégie
        trades = self.strategy.backtest()

        # Restaurer les données originales
        self.strategy.data = original_data

        # Calculer les métriques de performance
        performance = self._calculate_performance(trades)

        return performance

    def _calculate_performance(self, trades):
        """
        Calcule les métriques de performance

        Args:
            trades (list): Liste des trades

        Returns:
            dict: Métriques de performance
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "total_return": 0,
            }

        # Calculer les métriques de base
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade["profit"] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculer le profit factor
        gross_profit = sum(trade["profit"] for trade in trades if trade["profit"] > 0)
        gross_loss = abs(
            sum(trade["profit"] for trade in trades if trade["profit"] < 0)
        )
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calculer le rendement total
        total_return = sum(trade["profit"] for trade in trades)

        # Calculer le ratio de Sharpe (simplifié)
        returns = [trade["profit"] for trade in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # Calculer le drawdown maximum
        cumulative_returns = np.cumsum(returns)
        max_drawdown = 0
        peak = cumulative_returns[0]

        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
        }
