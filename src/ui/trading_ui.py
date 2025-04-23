#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interface utilisateur améliorée pour le bot de trading Bitcoin
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import MetaTrader5 as mt5
import logging
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use('TkAgg')

class TradingUI:
    """
    Interface utilisateur améliorée pour le bot de trading
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialise l'interface utilisateur
        
        Args:
            root: Fenêtre principale Tkinter
        """
        self.root = root
        self.root.title("Bitcoin Scalper Pro")
        self.root.geometry("1200x800")
        
        # Configuration du style
        self._setup_styles()
        
        # Création des widgets
        self._create_widgets()
        
        # Configuration des graphes
        self._setup_charts()
        
        # Configuration des tooltips
        self._setup_tooltips()
        
        # Queue pour la communication avec le bot
        self.message_queue = queue.Queue()
        
        # Initialisation de MT5
        self._init_mt5()
        
        # Flag pour contrôler la boucle de mise à jour
        self.is_running = True
        
        # ID de la tâche planifiée
        self.after_id = None
        
        # Démarrage de la boucle de mise à jour
        self._start_update_loop()
        
        # Configuration de la fermeture propre
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _setup_styles(self):
        """Configuration du style de l'interface"""
        style = ttk.Style()
        style.theme_use('clam')  # Utilisation du thème clam pour un look moderne
        
        # Configuration des couleurs
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', font=('Helvetica', 11, 'bold'))
        
        # Style pour les boutons
        style.configure('Connect.TButton', background='#4CAF50', foreground='white')
        style.configure('Disconnect.TButton', background='#f44336', foreground='white')
        
        # Style pour les labels d'état
        style.configure('Connected.TLabel', foreground='#4CAF50')
        style.configure('Disconnected.TLabel', foreground='#f44336')
        style.configure('Running.TLabel', foreground='#4CAF50')
        style.configure('Stopped.TLabel', foreground='#f44336')
        
    def _init_mt5(self):
        """Initialise la connexion à MT5"""
        try:
            if not mt5.initialize():
                self.log_message("Erreur: Impossible d'initialiser MT5")
                return False
                
            # Connexion au compte
            if not mt5.login(
                login=int(os.getenv('AVATRADE_LOGIN')),
                password=os.getenv('AVATRADE_PASSWORD'),
                server=os.getenv('AVATRADE_SERVER')
            ):
                self.log_message("Erreur: Impossible de se connecter au compte MT5")
                return False
                
            self.log_message("Connexion à MT5 réussie")
            return True
            
        except Exception as e:
            self.log_message(f"Erreur lors de l'initialisation de MT5: {str(e)}")
            return False
            
    def _create_widgets(self):
        """Création des widgets de l'interface"""
        # Frame principale avec padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame supérieure pour les informations de connexion et du compte
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)
        
        # Frame gauche pour l'état de la connexion et du bot
        status_frame = ttk.Frame(top_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Frame pour l'état de la connexion MT5
        connection_frame = ttk.LabelFrame(status_frame, text="État de la connexion", padding="10")
        connection_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.connection_label = ttk.Label(connection_frame, text="MT5: Non connecté", style='Disconnected.TLabel')
        self.connection_label.pack(side=tk.LEFT, padx=5)
        
        # Frame pour l'état du bot
        bot_status_frame = ttk.LabelFrame(status_frame, text="État du bot", padding="10")
        bot_status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.bot_status_label = ttk.Label(bot_status_frame, text="Bot: Arrêté", style='Disconnected.TLabel')
        self.bot_status_label.pack(side=tk.LEFT, padx=5)
        
        # Indicateur d'état du bot (cercle coloré)
        self.bot_status_indicator = ttk.Label(bot_status_frame, text="●", font=("Arial", 16))
        self.bot_status_indicator.pack(side=tk.LEFT, padx=5)
        self._update_bot_status_indicator(False)
        
        # Frame droite pour les informations du compte
        account_frame = ttk.LabelFrame(top_frame, text="Informations du compte", padding="10")
        account_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Labels pour les informations du compte
        self.balance_label = ttk.Label(account_frame, text="Balance: --")
        self.balance_label.pack(side=tk.LEFT, padx=5)
        
        self.equity_label = ttk.Label(account_frame, text="Équité: --")
        self.equity_label.pack(side=tk.LEFT, padx=5)
        
        self.profit_label = ttk.Label(account_frame, text="Profit: --")
        self.profit_label.pack(side=tk.LEFT, padx=5)
        
        # Frame centrale pour les graphiques
        chart_frame = ttk.LabelFrame(main_frame, text="Graphiques", padding="10")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Création des graphiques
        self._create_charts(chart_frame)
        
        # Frame inférieure pour les logs, positions et contrôles
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame gauche pour les logs, positions et historique
        info_frame = ttk.Frame(bottom_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Notebook pour les différents types d'informations
        self.info_notebook = ttk.Notebook(info_frame)
        self.info_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglet pour les logs généraux
        general_log_frame = ttk.Frame(self.info_notebook)
        self.info_notebook.add(general_log_frame, text="Logs Généraux")
        
        self.log_text = scrolledtext.ScrolledText(general_log_frame, height=10, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Onglet pour les événements de trading
        trade_log_frame = ttk.Frame(self.info_notebook)
        self.info_notebook.add(trade_log_frame, text="Événements Trading")
        
        self.trade_log_text = scrolledtext.ScrolledText(trade_log_frame, height=10, width=80)
        self.trade_log_text.pack(fill=tk.BOTH, expand=True)
        
        # Onglet pour les positions ouvertes
        positions_frame = ttk.Frame(self.info_notebook)
        self.info_notebook.add(positions_frame, text="Positions Ouvertes")
        
        # Tableau des positions ouvertes
        self.positions_tree = ttk.Treeview(positions_frame, columns=("symbol", "type", "volume", "price", "sl", "tp", "profit"), show="headings")
        self.positions_tree.heading("symbol", text="Symbole")
        self.positions_tree.heading("type", text="Type")
        self.positions_tree.heading("volume", text="Volume")
        self.positions_tree.heading("price", text="Prix")
        self.positions_tree.heading("sl", text="Stop Loss")
        self.positions_tree.heading("tp", text="Take Profit")
        self.positions_tree.heading("profit", text="Profit")
        self.positions_tree.pack(fill=tk.BOTH, expand=True)
        
        # Onglet pour l'historique des trades
        history_frame = ttk.Frame(self.info_notebook)
        self.info_notebook.add(history_frame, text="Historique des Trades")
        
        # Tableau de l'historique des trades
        self.history_tree = ttk.Treeview(history_frame, columns=("date", "symbol", "type", "volume", "price", "profit"), show="headings")
        self.history_tree.heading("date", text="Date")
        self.history_tree.heading("symbol", text="Symbole")
        self.history_tree.heading("type", text="Type")
        self.history_tree.heading("volume", text="Volume")
        self.history_tree.heading("price", text="Prix")
        self.history_tree.heading("profit", text="Profit")
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configuration des tags de couleur pour les logs
        self.log_text.tag_configure("INFO", foreground="black")
        self.log_text.tag_configure("WARNING", foreground="orange")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("SUCCESS", foreground="green")
        
        self.trade_log_text.tag_configure("OPEN", foreground="green")
        self.trade_log_text.tag_configure("CLOSE", foreground="red")
        self.trade_log_text.tag_configure("MODIFY", foreground="blue")
        self.trade_log_text.tag_configure("DELETE", foreground="orange")
        
        # Contrôles
        control_frame = ttk.Frame(bottom_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        self.start_button = ttk.Button(control_frame, text="Démarrer", command=self.start_bot)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(control_frame, text="Arrêter", command=self.stop_bot)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        # Bouton Quitter
        self.quit_button = ttk.Button(control_frame, text="Quitter", command=self._on_closing)
        self.quit_button.pack(fill=tk.X, pady=2)
        
        # Barre de statut
        self.status_bar = ttk.Label(main_frame, text="Prêt", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configuration des tooltips
        self._setup_tooltips()
        
    def _setup_charts(self):
        """Configure les graphiques"""
        # Configuration du graphique des prix
        self.price_ax.set_title('Prix du Bitcoin (USDT)')
        self.price_ax.set_xlabel('Temps')
        self.price_ax.set_ylabel('Prix (USDT)')
        self.price_ax.grid(True)
        self.price_ax.legend()
        
        # Configuration du graphique RSI
        self.rsi_ax.set_title('RSI (14)')
        self.rsi_ax.set_xlabel('Temps')
        self.rsi_ax.set_ylabel('RSI')
        self.rsi_ax.grid(True)
        self.rsi_ax.legend()
        self.rsi_ax.axhline(y=70, color='r', linestyle='--')
        self.rsi_ax.axhline(y=30, color='g', linestyle='--')
        
        # Configuration du graphique MACD
        self.macd_ax.set_title('MACD')
        self.macd_ax.set_xlabel('Temps')
        self.macd_ax.set_ylabel('MACD')
        self.macd_ax.grid(True)
        self.macd_ax.legend()
        
        # Ajustement de la mise en page
        plt.tight_layout()
        
    def _setup_tooltips(self):
        """Configure les tooltips pour les widgets"""
        self.create_tooltip(self.balance_label, "Solde actuel du compte")
        self.create_tooltip(self.equity_label, "Valeur totale du compte")
        self.create_tooltip(self.profit_label, "Profit/Perte actuel")
        self.create_tooltip(self.start_button, "Démarrer le bot de trading")
        self.create_tooltip(self.stop_button, "Arrêter le bot de trading")
        
    def create_tooltip(self, widget, text):
        """Crée un tooltip pour un widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
                
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            
        widget.bind('<Enter>', show_tooltip)
        
    def _start_update_loop(self):
        """Démarre la boucle de mise à jour de l'interface"""
        if not self.is_running:
            return
            
        try:
            self._process_messages()
            # Mise à jour des informations du compte toutes les secondes
            self._update_account_info(None)
        except Exception as e:
            self.log_message(f"Erreur dans la boucle de mise à jour: {str(e)}")
        finally:
            if self.is_running:
                self.after_id = self.root.after(1000, self._start_update_loop)  # Mise à jour toutes les secondes
                
    def stop_update_loop(self):
        """Arrête la boucle de mise à jour"""
        self.is_running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
            
    def _process_messages(self):
        """Traite les messages reçus du bot"""
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
            
    def _handle_message(self, message):
        """Gère un message reçu du bot"""
        try:
            message_type = message.get("type")
            data = message.get("data")
            
            self._add_log_message(f"Message reçu: {message_type}")
            
            if message_type == "market_data":
                self._update_price_chart(data)
            elif message_type == "indicators":
                self._update_indicator_charts(data)
            elif message_type == "account_info":
                self._update_account_info(data)
            elif message_type == "log":
                if data is None:
                    # Ignorer les messages sans contenu
                    pass
                elif isinstance(data, str):
                    if data.strip():  # Vérifier si le message n'est pas vide après suppression des espaces
                        self._add_log_message(data)
                elif isinstance(data, dict):
                    log_message = data.get("message", "")
                    level = data.get("level", "INFO")
                    if log_message.strip():  # Vérifier si le message n'est pas vide après suppression des espaces
                        self._add_log_message(log_message, level)
                else:
                    self._add_log_message(f"Format de log invalide: {type(data)}", "ERROR")
            elif message_type == "trade":
                if isinstance(data, dict):
                    action = data.get("action")
                    symbol = data.get("symbol", "BTCUSDT")
                    volume = data.get("volume", 0)
                    price = data.get("price", 0)
                    order_type = data.get("order_type", "")
                    position_id = data.get("position_id", "")
                    
                    if action == "open":
                        message = f"Position ouverte: {symbol} - {volume} lots à {price} USDT ({order_type})"
                        self._add_trade_message(message, "OPEN")
                        # Ajouter à la liste des positions ouvertes
                        self.positions_tree.insert("", "end", values=(symbol, order_type, volume, price, data.get("sl", "N/A"), data.get("tp", "N/A"), "0.00"))
                    elif action == "close":
                        message = f"Position fermée: {symbol} - {volume} lots à {price} USDT (ID: {position_id})"
                        self._add_trade_message(message, "CLOSE")
                        # Retirer de la liste des positions ouvertes
                        for item in self.positions_tree.get_children():
                            if self.positions_tree.item(item)["values"][0] == symbol and self.positions_tree.item(item)["values"][1] == order_type:
                                self.positions_tree.delete(item)
                                break
                        # Ajouter à l'historique des trades
                        profit = data.get("profit", 0)
                        self.history_tree.insert("", 0, values=(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, order_type, volume, price, profit))
                    elif action == "modify":
                        message = f"Position modifiée: {symbol} - Nouveau SL: {data.get('sl', 'N/A')} USDT, Nouveau TP: {data.get('tp', 'N/A')} USDT"
                        self._add_trade_message(message, "MODIFY")
                        # Mettre à jour la position dans la liste
                        for item in self.positions_tree.get_children():
                            if self.positions_tree.item(item)["values"][0] == symbol and self.positions_tree.item(item)["values"][1] == order_type:
                                values = list(self.positions_tree.item(item)["values"])
                                values[4] = data.get("sl", "N/A")
                                values[5] = data.get("tp", "N/A")
                                self.positions_tree.item(item, values=values)
                                break
                    elif action == "delete":
                        message = f"Ordre supprimé: {symbol} - Type: {order_type}"
                        self._add_trade_message(message, "DELETE")
                        # Retirer de la liste des positions ouvertes
                        for item in self.positions_tree.get_children():
                            if self.positions_tree.item(item)["values"][0] == symbol and self.positions_tree.item(item)["values"][1] == order_type:
                                self.positions_tree.delete(item)
                                break
                    else:
                        self._add_log_message(f"Action de trading inconnue: {action}", "WARNING")
                else:
                    self._add_log_message("Format de message de trading invalide", "ERROR")
            elif message_type == "positions":
                # Mise à jour de la liste des positions ouvertes
                if isinstance(data, list):
                    # Effacer la liste actuelle
                    for item in self.positions_tree.get_children():
                        self.positions_tree.delete(item)
                    # Ajouter les nouvelles positions
                    for position in data:
                        self.positions_tree.insert("", "end", values=(
                            position.get("symbol", ""),
                            position.get("type", ""),
                            position.get("volume", 0),
                            position.get("price", 0),
                            position.get("sl", "N/A"),
                            position.get("tp", "N/A"),
                            position.get("profit", "0.00")
                        ))
            elif message_type == "history":
                # Mise à jour de l'historique des trades
                if isinstance(data, list):
                    # Effacer l'historique actuel
                    for item in self.history_tree.get_children():
                        self.history_tree.delete(item)
                    # Ajouter les nouveaux trades
                    for trade in data:
                        self.history_tree.insert("", 0, values=(
                            trade.get("date", ""),
                            trade.get("symbol", ""),
                            trade.get("type", ""),
                            trade.get("volume", 0),
                            trade.get("price", 0),
                            trade.get("profit", "0.00")
                        ))
        except Exception as e:
            self._add_log_message(f"Erreur lors du traitement du message: {str(e)}", "ERROR")
        
    def _update_price_chart(self, data):
        """Met à jour le graphe des prix"""
        try:
            # Vérification que les données sont dans le bon format
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    prices = data['close'].values
                    self.log_message(f"Mise à jour du graphe avec {len(prices)} points de données")
                elif 'bid' in data.columns:
                    prices = data['bid'].values
                    self.log_message(f"Mise à jour du graphe avec {len(prices)} points de données")
                else:
                    self.log_message("Erreur: Colonnes disponibles: " + ", ".join(data.columns))
                    return
            else:
                self.log_message(f"Erreur: Type de données reçu: {type(data)}")
                return
            
            # Mise à jour des données
            self.price_data = prices
            self.price_line.set_data(range(len(prices)), prices)
            self.price_ax.relim()
            self.price_ax.autoscale_view()
            
            # Mise à jour de l'affichage
            self.price_canvas.draw()
            
        except Exception as e:
            self.log_message(f"Erreur lors de la mise à jour du graphe des prix: {str(e)}")
        
    def _update_indicator_charts(self, data):
        """Met à jour les graphes des indicateurs"""
        # Mise à jour RSI
        self.rsi_line.set_data(range(len(self.rsi_data)), self.rsi_data)
        self.rsi_ax.relim()
        self.rsi_ax.autoscale_view()
        self.rsi_canvas.draw()
        
        # Mise à jour MACD
        self.macd_line.set_data(range(len(self.macd_data)), self.macd_data)
        self.signal_line.set_data(range(len(self.signal_data)), self.signal_data)
        self.macd_ax.relim()
        self.macd_ax.autoscale_view()
        self.macd_canvas.draw()
        
    def _update_account_info(self, data):
        """Met à jour les informations du compte"""
        try:
            if not mt5.initialize():
                self.log_message("Erreur: MT5 non initialisé")
                return
                
            account_info = mt5.account_info()
            if account_info is None:
                self.log_message("Erreur: Impossible de récupérer les informations du compte")
                return
                
            # Mise à jour des labels avec les informations du compte
            self.balance_label.config(text=f"Balance: {account_info.balance:.2f} USDT")
            self.equity_label.config(text=f"Équité: {account_info.equity:.2f} USDT")
            self.profit_label.config(text=f"Profit: {account_info.profit:.2f} USDT")
            
        except Exception as e:
            self.log_message(f"Erreur lors de la mise à jour des informations du compte: {str(e)}")
        
    def _add_log_message(self, message, level="INFO"):
        """Ajoute un message au log avec un niveau spécifique"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n", level)
        self.log_text.see("end")
        
    def _add_trade_message(self, message, trade_type):
        """Ajoute un message de trading dans l'onglet dédié"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.trade_log_text.insert("end", f"[{timestamp}] {message}\n", trade_type)
        self.trade_log_text.see("end")
        
    def _on_closing(self):
        """Gère la fermeture propre de l'application"""
        try:
            # Arrêt de la boucle de mise à jour
            self.stop_update_loop()
            
            # Arrêt du bot si en cours d'exécution
            if hasattr(self, 'stop_bot_callback'):
                self.stop_bot_callback()
                self.log_message("Bot arrêté avec succès")
            
            # Déconnexion de MT5
            if mt5.initialize():
                mt5.shutdown()
                
            # Fermeture de la fenêtre
            self.root.destroy()
            
            # Arrêt du processus Python
            import sys
            sys.exit(0)
            
        except Exception as e:
            print(f"Erreur lors de la fermeture de l'application: {str(e)}")
            self.root.destroy()
            import sys
            sys.exit(1)
        
    def log_message(self, message):
        """Ajoute un message au log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        logging.info(message)
        
    def _update_bot_status_indicator(self, is_running):
        """Met à jour l'indicateur d'état du bot"""
        if is_running:
            self.bot_status_indicator.configure(foreground='#4CAF50')  # Vert
            self.bot_status_label.configure(text="Bot: En cours d'exécution", style='Running.TLabel')
        else:
            self.bot_status_indicator.configure(foreground='#f44336')  # Rouge
            self.bot_status_label.configure(text="Bot: Arrêté", style='Stopped.TLabel')
            
    def start_bot(self):
        """Démarre le bot de trading"""
        try:
            # Vérification de la connexion MT5
            if not mt5.initialize() or not mt5.terminal_info():
                if not self._init_mt5():
                    return
                
            # Mise à jour de l'interface
            self.connection_label.config(text="MT5: Connecté", style='Connected.TLabel')
            self.status_bar.config(text="Bot en cours d'exécution")
            self._update_bot_status_indicator(True)
            
            # Démarrage du bot via le callback
            if hasattr(self, 'start_bot_callback'):
                self.start_bot_callback()
                self.log_message("Bot démarré avec succès")
            else:
                self.log_message("Erreur: Callback de démarrage non configuré")
                
        except Exception as e:
            self.log_message(f"Erreur lors du démarrage du bot: {str(e)}")
            
    def stop_bot(self):
        """Arrête le bot de trading"""
        try:
            # Arrêt du bot via le callback
            if hasattr(self, 'stop_bot_callback'):
                self.stop_bot_callback()
                self.log_message("Bot arrêté avec succès")
            else:
                self.log_message("Erreur: Callback d'arrêt non configuré")
                
            # Mise à jour de l'interface
            self.status_bar.config(text="Bot arrêté")
            self._update_bot_status_indicator(False)
            
        except Exception as e:
            self.log_message(f"Erreur lors de l'arrêt du bot: {str(e)}")
        
    def _create_charts(self, parent):
        """Crée les graphiques pour l'interface"""
        # Frame pour les graphiques
        charts_container = ttk.Frame(parent)
        charts_container.pack(fill=tk.BOTH, expand=True)
        
        # Graphique des prix
        self.price_fig, self.price_ax = plt.subplots(figsize=(6, 4))
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, master=charts_container)
        self.price_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Graphique RSI
        self.rsi_fig, self.rsi_ax = plt.subplots(figsize=(6, 4))
        self.rsi_canvas = FigureCanvasTkAgg(self.rsi_fig, master=charts_container)
        self.rsi_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Graphique MACD
        self.macd_fig, self.macd_ax = plt.subplots(figsize=(6, 4))
        self.macd_canvas = FigureCanvasTkAgg(self.macd_fig, master=charts_container)
        self.macd_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Initialisation des données
        self.price_data = []
        self.rsi_data = []
        self.macd_data = []
        self.signal_data = []
        
        # Création des lignes
        self.price_line, = self.price_ax.plot([], [], label='Prix')
        self.rsi_line, = self.rsi_ax.plot([], [], label='RSI')
        self.macd_line, = self.macd_ax.plot([], [], label='MACD')
        self.signal_line, = self.macd_ax.plot([], [], label='Signal')
        
        # Configuration des graphiques
        self._setup_charts() 