import os
import sys
import time
import logging

logger = logging.getLogger(__name__)

def monitor_thread_health():
    while True:
        if not refresh_manager._thread.is_alive():
            logger.critical("Thread principal crashé - Redémarrage...")
            os.execv(sys.executable, ['python'] + sys.argv)
        time.sleep(5)
