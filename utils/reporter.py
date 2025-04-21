#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de génération de rapports pour le bot de scalping
Génère des rapports quotidiens et des résumés de performance
"""

import os
import csv
import json
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

class TradingReporter:
    """
    Classe pour générer des rapports et journaliser les trades
    """
    
    def __init__(self, log_dir="reports"):
        """
        Initialisation du reporter
        
        Args:
            log_dir (str): Répertoire pour les rapports
        """
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir
        
        # Création du répertoire de rapports si nécessaire
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Chemins des fichiers
        self.trades_file = os.path.join(log_dir, "trades.csv")
        self.daily_report_file = os.path.join(log_dir, "daily_report_{date}.txt")
        self.summary_report_file = os.path.join(log_dir, "performance_summary.json")
        
        # Historique de trades
        self.trades = []
        
        # Chargement des trades historiques
        self._load_trades()
        
        self.logger.info("Reporter initialisé dans le répertoire %s", log_dir)
    
    def _load_trades(self):
        """
        Charge l'historique des trades depuis le fichier CSV
        """
        if not os.path.exists(self.trades_file):
            # Création du fichier avec l'en-tête
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'order_id', 'type', 'volume', 'price', 
                    'sl', 'tp', 'close_price', 'close_time', 'profit', 'pips'
                ])
            return
        
        try:
            # Chargement des trades
            df = pd.read_csv(self.trades_file)
            self.trades = df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des trades: {e}")
    
    def log_trade(self, signal, result):
        """
        Enregistre un nouveau trade dans l'historique
        
        Args:
            signal (dict): Signal qui a généré le trade
            result (dict): Résultat de l'exécution de l'ordre
        """
        trade = {
            'timestamp': result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'order_id': result.get('order_id', ''),
            'type': signal.get('type', ''),
            'volume': result.get('volume', 0),
            'price': result.get('price', 0),
            'sl': signal.get('sl', 0),
            'tp': signal.get('tp', 0),
            'close_price': None,
            'close_time': None,
            'profit': None,
            'pips': None
        }
        
        # Ajout à l'historique
        self.trades.append(trade)
        
        # Écriture dans le fichier CSV
        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade['timestamp'], 
                trade['order_id'], 
                trade['type'], 
                trade['volume'], 
                trade['price'], 
                trade['sl'], 
                trade['tp'], 
                trade['close_price'], 
                trade['close_time'], 
                trade['profit'], 
                trade['pips']
            ])
        
        self.logger.info(f"Trade enregistré: {trade['type']} à {trade['price']}")
    
    def update_trade_result(self, order_id, close_price, close_time, profit):
        """
        Met à jour un trade avec son résultat
        
        Args:
            order_id (str): ID de l'ordre
            close_price (float): Prix de clôture
            close_time (str): Heure de clôture
            profit (float): Profit/perte
        """
        # Recherche du trade dans l'historique
        for trade in self.trades:
            if trade['order_id'] == order_id:
                trade['close_price'] = close_price
                trade['close_time'] = close_time
                trade['profit'] = profit
                
                # Calcul des pips
                if trade['price'] and close_price:
                    # Pour BTC, 1 pip = 0.0001
                    pips = (close_price - trade['price']) / 0.0001
                    if trade['type'] == 'sell':
                        pips = -pips
                    trade['pips'] = pips
                
                # Mise à jour du fichier CSV
                df = pd.DataFrame(self.trades)
                df.to_csv(self.trades_file, index=False)
                
                self.logger.info(f"Trade mis à jour: {order_id} avec profit {profit}")
                return
        
        self.logger.warning(f"Trade non trouvé pour mise à jour: {order_id}")
    
    def generate_daily_report(self, date=None):
        """
        Génère un rapport quotidien
        
        Args:
            date (str): Date du rapport (format YYYY-MM-DD) ou None pour aujourd'hui
            
        Returns:
            str: Chemin du rapport généré
        """
        # Date par défaut = aujourd'hui
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Filtrage des trades du jour
        day_trades = [
            t for t in self.trades 
            if t['timestamp'].startswith(date) or 
               (t['close_time'] and t['close_time'].startswith(date))
        ]
        
        if not day_trades:
            self.logger.info(f"Aucun trade pour le rapport du {date}")
            return None
        
        # Nom du fichier
        report_file = self.daily_report_file.format(date=date)
        
        try:
            with open(report_file, 'w') as f:
                f.write(f"=== Rapport de Trading du {date} ===\n\n")
                
                # Statistiques générales
                total_trades = len(day_trades)
                closed_trades = sum(1 for t in day_trades if t['close_price'] is not None)
                
                total_profit = sum(t['profit'] for t in day_trades if t['profit'] is not None)
                winning_trades = sum(1 for t in day_trades if t['profit'] is not None and t['profit'] > 0)
                
                if closed_trades > 0:
                    win_rate = winning_trades / closed_trades * 100
                else:
                    win_rate = 0
                
                f.write(f"Nombre total de trades: {total_trades}\n")
                f.write(f"Trades clôturés: {closed_trades}\n")
                f.write(f"Trades gagnants: {winning_trades}\n")
                f.write(f"Win rate: {win_rate:.2f}%\n")
                f.write(f"Profit total: {total_profit:.6f} BTC\n\n")
                
                # Détail des trades
                f.write("=== Détail des Trades ===\n")
                for trade in day_trades:
                    f.write(f"\nOrder ID: {trade['order_id']}\n")
                    f.write(f"Type: {trade['type']}\n")
                    f.write(f"Volume: {trade['volume']}\n")
                    f.write(f"Prix d'entrée: {trade['price']}\n")
                    f.write(f"Stop Loss: {trade['sl']}\n")
                    f.write(f"Take Profit: {trade['tp']}\n")
                    
                    if trade['close_price'] is not None:
                        f.write(f"Prix de sortie: {trade['close_price']}\n")
                        f.write(f"Heure de sortie: {trade['close_time']}\n")
                        f.write(f"Profit: {trade['profit']}\n")
                        f.write(f"Pips: {trade['pips']}\n")
            
            # Génération d'un graphique
            try:
                self._generate_daily_chart(day_trades, date)
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération du graphique: {e}")
            
            self.logger.info(f"Rapport quotidien généré: {report_file}")
            return report_file
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport quotidien: {e}")
            return None
    
    def _generate_daily_chart(self, day_trades, date):
        """
        Génère un graphique de performance quotidienne
        
        Args:
            day_trades (list): Liste des trades du jour
            date (str): Date du rapport
        """
        # Création d'un DataFrame pour le graphique
        closed_trades = [t for t in day_trades if t['close_price'] is not None and t['profit'] is not None]
        
        if not closed_trades:
            return
        
        # Tri par heure de clôture
        df = pd.DataFrame(closed_trades)
        df['close_time'] = pd.to_datetime(df['close_time'])
        df = df.sort_values('close_time')
        
        # Calcul du profit cumulé
        df['cumulative_profit'] = df['profit'].cumsum()
        
        # Génération du graphique
        plt.figure(figsize=(10, 6))
        plt.plot(df['close_time'], df['cumulative_profit'], marker='o', linestyle='-')
        plt.title(f'Performance de Trading - {date}')
        plt.xlabel('Heure')
        plt.ylabel('Profit cumulé (BTC)')
        plt.grid(True)
        
        # Sauvegarde du graphique
        chart_file = os.path.join(self.log_dir, f"daily_chart_{date}.png")
        plt.savefig(chart_file)
        plt.close()
    
    def generate_summary_report(self):
        """
        Génère un rapport de synthèse global des performances
        
        Returns:
            str: Chemin du rapport généré
        """
        try:
            # Filtrage des trades clôturés
            closed_trades = [t for t in self.trades if t['close_price'] is not None and t['profit'] is not None]
            
            if not closed_trades:
                self.logger.info("Aucun trade clôturé pour le rapport de synthèse")
                return None
            
            # Calcul des statistiques
            total_trades = len(closed_trades)
            winning_trades = sum(1 for t in closed_trades if t['profit'] > 0)
            losing_trades = sum(1 for t in closed_trades if t['profit'] < 0)
            
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            total_profit = sum(t['profit'] for t in closed_trades)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            winning_profits = [t['profit'] for t in closed_trades if t['profit'] > 0]
            avg_win = sum(winning_profits) / len(winning_profits) if winning_profits else 0
            
            losing_profits = [t['profit'] for t in closed_trades if t['profit'] < 0]
            avg_loss = sum(losing_profits) / len(losing_profits) if losing_profits else 0
            
            profit_factor = abs(sum(winning_profits) / sum(losing_profits)) if sum(losing_profits) != 0 else 0
            
            # Calcul du drawdown max
            df = pd.DataFrame(closed_trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df['cumulative_profit'] = df['profit'].cumsum()
            
            max_drawdown = 0
            peak = 0
            
            for profit in df['cumulative_profit']:
                if profit > peak:
                    peak = profit
                drawdown = peak - profit
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Création du rapport JSON
            summary = {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'average_profit': avg_profit,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'best_trade': max(t['profit'] for t in closed_trades) if closed_trades else 0,
                'worst_trade': min(t['profit'] for t in closed_trades) if closed_trades else 0,
                'trade_distribution': {
                    'buy': sum(1 for t in closed_trades if t['type'] == 'buy'),
                    'sell': sum(1 for t in closed_trades if t['type'] == 'sell')
                }
            }
            
            # Sauvegarde du rapport
            with open(self.summary_report_file, 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Génération d'un graphique de performance
            try:
                self._generate_summary_chart(df)
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération du graphique de synthèse: {e}")
            
            self.logger.info(f"Rapport de synthèse généré: {self.summary_report_file}")
            return self.summary_report_file
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport de synthèse: {e}")
            return None

    def _generate_summary_chart(self, df):
        """
        Génère un graphique de synthèse des performances
        
        Args:
            df (pd.DataFrame): DataFrame des trades avec profit cumulé
        """
        # Génération de graphiques
        plt.figure(figsize=(12, 8))
        
        # Graphique de l'évolution du profit
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['cumulative_profit'], marker='', linestyle='-')
        plt.title('Évolution du Profit Cumulé')
        plt.grid(True)
        
        # Graphique des profits par trade
        plt.subplot(2, 1, 2)
        colors = ['green' if p > 0 else 'red' for p in df['profit']]
        plt.bar(range(len(df)), df['profit'], color=colors)
        plt.title('Profit par Trade')
        plt.xlabel('Numéro de Trade')
        plt.ylabel('Profit (BTC)')
        plt.grid(True)
        
        # Ajustement du layout
        plt.tight_layout()
        
        # Sauvegarde du graphique
        chart_file = os.path.join(self.log_dir, "performance_summary.png")
        plt.savefig(chart_file)
        plt.close()