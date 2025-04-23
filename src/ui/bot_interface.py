#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
import threading
import queue
import logging
from datetime import datetime

class BotInterface:
    """Interface graphique pour le bot de trading"""
    
    def __init__(self, bot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("Bitcoin Scalper Bot")
        self.root.geometry("800x600")
        
        # Queue pour la communication entre le bot et l'interface
        self.log_queue = queue.Queue()
        
        # État de connexion
        self.connection_status = False
        
        self.setup_ui()
        self.setup_logging()
        
        # Ajout d'un gestionnaire d'erreurs global
        self.root.report_callback_exception = self.handle_error
        
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Informations du compte
        account_frame = ttk.LabelFrame(main_frame, text="Informations du compte", padding="5")
        account_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.balance_label = ttk.Label(account_frame, text="Balance: --")
        self.balance_label.grid(row=0, column=0, padx=5)
        
        self.equity_label = ttk.Label(account_frame, text="Équité: --")
        self.equity_label.grid(row=0, column=1, padx=5)
        
        # État du bot
        status_frame = ttk.LabelFrame(main_frame, text="État du bot", padding="5")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(status_frame, text="État: En attente")
        self.status_label.grid(row=0, column=0, padx=5)
        
        # Dernier signal
        signal_frame = ttk.LabelFrame(main_frame, text="Dernier signal", padding="5")
        signal_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.signal_label = ttk.Label(signal_frame, text="Signal: --")
        self.signal_label.grid(row=0, column=0, padx=5)
        
        # Logs
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="5")
        log_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text['yscrollcommand'] = scrollbar.set
        
        # Boutons de contrôle
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=2, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Démarrer", command=self.start_bot)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Arrêter", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Configuration du redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
    def setup_logging(self):
        """Configuration du logging pour l'interface"""
        class QueueHandler(logging.Handler):
            def __init__(self, queue):
                super().__init__()
                self.queue = queue
                
            def emit(self, record):
                self.queue.put(record)
                
        # Ajout du handler pour l'interface
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Configuration du logger root
        root_logger = logging.getLogger()
        root_logger.addHandler(queue_handler)
        
    def handle_error(self, exc_type, exc_value, exc_traceback):
        """Gestionnaire d'erreurs global pour l'interface"""
        error_msg = f"Erreur: {exc_type.__name__}: {str(exc_value)}"
        logging.error(error_msg)
        self.log_text.insert(tk.END, f"\n{error_msg}\n")
        self.log_text.see(tk.END)
        
    def update_ui(self):
        """Mise à jour de l'interface"""
        try:
            # Mise à jour des logs
            while True:
                record = self.log_queue.get_nowait()
                msg = self.log_text.get("1.0", tk.END)
                self.log_text.delete("1.0", tk.END)
                self.log_text.insert("1.0", msg + self.format_log(record))
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des logs: {str(e)}")
            
        try:
            # Mise à jour des informations du compte
            if hasattr(self.bot, 'connector') and self.bot.connector:
                if self.bot.connector.is_connected:
                    account_info = self.bot.connector.get_account_info()
                    if account_info:
                        self.balance_label.config(text=f"Balance: {account_info['balance']:.2f}")
                        self.equity_label.config(text=f"Équité: {account_info['equity']:.2f}")
                    else:
                        self.balance_label.config(text="Balance: --")
                        self.equity_label.config(text="Équité: --")
                else:
                    self.status_label.config(text="État: Déconnecté")
                    self.balance_label.config(text="Balance: --")
                    self.equity_label.config(text="Équité: --")
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des informations du compte: {str(e)}")
                
        # Planification de la prochaine mise à jour
        self.root.after(1000, self.update_ui)
        
    def format_log(self, record):
        """Formatage des messages de log"""
        return f"{datetime.fromtimestamp(record.created).strftime('%H:%M:%S')} - {record.levelname} - {record.getMessage()}\n"
        
    def start_bot(self):
        """Démarrage du bot dans un thread séparé"""
        try:
            logging.info("Démarrage du bot...")
            self.bot_thread = threading.Thread(target=self._run_bot)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="État: En cours d'exécution")
        except Exception as e:
            logging.error(f"Erreur lors du démarrage du bot: {str(e)}")
            self.status_label.config(text="État: Erreur de démarrage")
            
    def _run_bot(self):
        """Exécution du bot avec gestion des erreurs"""
        try:
            self.bot.run()
        except Exception as e:
            logging.error(f"Erreur dans le thread du bot: {str(e)}")
            self.root.after(0, lambda: self.status_label.config(text="État: Erreur d'exécution"))
            
    def stop_bot(self):
        """Arrêt du bot"""
        try:
            logging.info("Arrêt du bot...")
            self.bot.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="État: Arrêté")
        except Exception as e:
            logging.error(f"Erreur lors de l'arrêt du bot: {str(e)}")
            self.status_label.config(text="État: Erreur d'arrêt")
        
    def run(self):
        """Lancement de l'interface"""
        self.update_ui()
        self.root.mainloop() 