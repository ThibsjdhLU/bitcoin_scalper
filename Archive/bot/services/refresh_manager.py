import threading

class RefreshManager:
    """Gère le rafraîchissement des données."""
    
    def __init__(self):
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        """Démarre le processus de rafraîchissement."""
        self.running = True
    
    def stop(self):
        """Arrête le processus de rafraîchissement."""
        self.running = False
