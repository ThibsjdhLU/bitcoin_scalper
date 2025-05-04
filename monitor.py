"""
Interface de monitoring CLI pour le bot de trading.
"""
import time
from datetime import datetime
from typing import Dict, List
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from loguru import logger

from core.risk_manager import RiskManager
from core.mt5_connector import MT5Connector

class Monitor:
    """Interface de monitoring en temps réel."""
    
    def __init__(
        self,
        risk_manager: RiskManager,
        mt5_connector: MT5Connector,
        refresh_rate: float = 1.0
    ):
        """
        Initialise le moniteur.
        
        Args:
            risk_manager: Gestionnaire de risque
            mt5_connector: Connecteur MT5
            refresh_rate: Taux de rafraîchissement en secondes
        """
        self.risk_manager = risk_manager
        self.mt5_connector = mt5_connector
        self.refresh_rate = refresh_rate
        self.console = Console()
        
    def generate_positions_table(self) -> Table:
        """
        Génère le tableau des positions ouvertes.
        
        Returns:
            Table: Tableau rich des positions
        """
        table = Table(
            title="Positions Ouvertes",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        # Colonnes
        table.add_column("Symbole", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Volume", justify="right")
        table.add_column("Prix Entrée", justify="right")
        table.add_column("Prix Actuel", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        
        # Données
        positions = self.risk_manager.open_positions
        for symbol, pos in positions.items():
            pnl = pos.get('unrealized_pnl', 0)
            pnl_color = "green" if pnl >= 0 else "red"
            
            table.add_row(
                symbol,
                pos.get('type', ''),
                f"{pos.get('volume', 0):.2f}",
                f"{pos.get('entry_price', 0):.2f}",
                f"{pos.get('current_price', 0):.2f}",
                f"[{pnl_color}]{pnl:.2f}[/{pnl_color}]",
                f"{pos.get('stop_loss', 0):.2f}",
                f"{pos.get('take_profit', 0):.2f}"
            )
            
        return table
        
    def generate_metrics_panel(self) -> Panel:
        """
        Génère le panneau des métriques.
        
        Returns:
            Panel: Panneau rich des métriques
        """
        metrics = self.risk_manager.get_risk_metrics()
        
        # Formater les métriques
        daily_pnl = metrics['daily_pnl']
        pnl_color = "green" if daily_pnl >= 0 else "red"
        drawdown = metrics['current_drawdown'] * 100
        
        content = [
            f"Capital: {metrics['current_capital']:.2f}",
            f"P&L Journalier: [{pnl_color}]{daily_pnl:.2f}[/{pnl_color}]",
            f"Drawdown: {drawdown:.2f}%",
            f"Trades Aujourd'hui: {metrics['daily_trades']}",
            f"Positions Ouvertes: {metrics['open_positions']}"
        ]
        
        return Panel(
            "\n".join(content),
            title="Métriques",
            border_style="blue"
        )
        
    def generate_alerts_panel(self) -> Panel:
        """
        Génère le panneau des alertes.
        
        Returns:
            Panel: Panneau rich des alertes
        """
        # TODO: Implémenter un système de queue d'alertes
        alerts = [
            "Aucune alerte importante"
        ]
        
        return Panel(
            "\n".join(alerts),
            title="Alertes",
            border_style="yellow"
        )
        
    def update(self) -> Layout:
        """
        Met à jour l'interface.
        
        Returns:
            Layout: Layout rich complet
        """
        layout = Layout()
        
        # Division verticale principale
        layout.split_column(
            Layout(name="upper"),
            Layout(name="lower")
        )
        
        # Division horizontale supérieure
        layout["upper"].split_row(
            Layout(self.generate_metrics_panel(), name="metrics", ratio=1),
            Layout(self.generate_alerts_panel(), name="alerts", ratio=1)
        )
        
        # Positions dans la partie inférieure
        layout["lower"].update(self.generate_positions_table())
        
        return layout
        
    def run(self):
        """Lance le moniteur en temps réel."""
        try:
            with Live(
                self.update(),
                refresh_per_second=1/self.refresh_rate,
                screen=True
            ) as live:
                while True:
                    live.update(self.update())
                    time.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            logger.info("Moniteur arrêté par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur du moniteur: {str(e)}")
            raise

def main(
    refresh_rate: float = typer.Option(
        1.0,
        help="Taux de rafraîchissement en secondes"
    )
):
    """Point d'entrée du moniteur CLI."""
    try:
        # Charger la configuration
        with open('config/risk_config.json', 'r') as f:
            risk_config = json.load(f)
        with open('config/mt5_config.json', 'r') as f:
            mt5_config = json.load(f)
            
        # Initialiser les composants
        risk_manager = RiskManager(risk_config)
        mt5_connector = MT5Connector(**mt5_config)
        
        # Lancer le moniteur
        monitor = Monitor(
            risk_manager=risk_manager,
            mt5_connector=mt5_connector,
            refresh_rate=refresh_rate
        )
        monitor.run()
        
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        raise

if __name__ == "__main__":
    typer.run(main) 