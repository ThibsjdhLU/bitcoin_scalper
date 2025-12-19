"""
✅ PHASE 5: INFERENCE & SAFETY (Le Live Trading)
Module de sécurité pour l'inférence en temps réel.
Implémente:
- Garde Latence (Latency Guard)
- Filtre d'Entropie (Le Doute)
- Risk Management Dynamique basé sur la confiance du modèle
- Monitoring des erreurs consécutives (Kill Switch)
"""
import time
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger("bitcoin_scalper.inference_safety")
logger.setLevel(logging.INFO)


class InferenceSafetyGuard:
    """
    Garde de sécurité pour l'inférence en temps réel.
    Protège le capital via des vérifications strictes avant chaque trade.
    """
    
    def __init__(
        self,
        max_latency_ms: float = 200.0,
        max_entropy: float = 0.8,
        max_consecutive_errors: int = 5,
        error_window_seconds: int = 60
    ):
        """
        Initialise le garde de sécurité.
        
        Args:
            max_latency_ms: Latence maximale acceptable en millisecondes (défaut: 200ms)
            max_entropy: Entropie maximale acceptable pour les probabilités (défaut: 0.8)
            max_consecutive_errors: Nombre d'erreurs consécutives avant kill switch (défaut: 5)
            error_window_seconds: Fenêtre temporelle pour compter les erreurs (défaut: 60s)
        """
        self.max_latency_ms = max_latency_ms
        self.max_entropy = max_entropy
        self.max_consecutive_errors = max_consecutive_errors
        self.error_window_seconds = error_window_seconds
        
        # Error tracking for kill switch
        self.error_timestamps = deque(maxlen=max_consecutive_errors * 2)
        self.consecutive_errors = 0
        self.kill_switch_active = False
        
        # Statistics
        self.total_checks = 0
        self.latency_rejects = 0
        self.entropy_rejects = 0
        self.kill_switch_triggers = 0
    
    def check_latency(self, tick_timestamp: datetime) -> Tuple[bool, str, float]:
        """
        ✅ Garde Latence (Latency Guard)
        
        Vérifie que la latence entre la réception du tick et maintenant est acceptable.
        
        Règle: Si latence > 200ms → ABORT TRADE
        
        Args:
            tick_timestamp: Timestamp du tick de données reçu
            
        Returns:
            (passed: bool, reason: str, latency_ms: float)
        """
        now = datetime.now(tz=tick_timestamp.tzinfo) if tick_timestamp.tzinfo else datetime.now()
        delta = now - tick_timestamp
        latency_ms = delta.total_seconds() * 1000
        
        if latency_ms > self.max_latency_ms:
            self.latency_rejects += 1
            reason = f"⛔ LATENCY GUARD: {latency_ms:.1f}ms > {self.max_latency_ms}ms - ABORT TRADE"
            logger.warning(reason)
            return False, reason, latency_ms
        
        logger.debug(f"✅ Latency OK: {latency_ms:.1f}ms")
        return True, "Latency OK", latency_ms
    
    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        ✅ Filtre d'Entropie (Le Doute)
        
        Calcule l'entropie de Shannon sur les probabilités de sortie.
        H(X) = -Σ p(x) * log2(p(x))
        
        Pour N classes uniformément distribuées: H_max = log2(N)
        - 2 classes: H_max = 1.0
        - 3 classes: H_max = 1.585
        
        Args:
            probabilities: Array de probabilités (doit sommer à 1.0)
            
        Returns:
            entropy: Valeur d'entropie (0.0 = certitude parfaite, H_max = confusion maximale)
        """
        # Normalisation et clip pour éviter log(0)
        probs = np.array(probabilities, dtype=float)
        probs = np.clip(probs, 1e-9, 1.0)
        probs = probs / probs.sum()
        
        # Calcul de l'entropie
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy
    
    def check_entropy(self, probabilities: np.ndarray) -> Tuple[bool, str, float]:
        """
        ✅ Filtre d'Entropie (Le Doute)
        
        Vérifie que le modèle n'est pas confus (entropie trop élevée).
        
        Règle: Si Entropie > Seuil (ex: 0.8) → NO TRADE (Modèle confus)
        
        Args:
            probabilities: Array de probabilités de sortie du modèle
            
        Returns:
            (passed: bool, reason: str, entropy: float)
        """
        entropy = self.calculate_entropy(probabilities)
        
        if entropy > self.max_entropy:
            self.entropy_rejects += 1
            reason = f"⛔ ENTROPY FILTER: {entropy:.3f} > {self.max_entropy} - NO TRADE (Modèle confus)"
            logger.warning(reason)
            return False, reason, entropy
        
        logger.debug(f"✅ Entropy OK: {entropy:.3f}")
        return True, "Entropy OK", entropy
    
    def record_error(self):
        """
        Enregistre une erreur pour le suivi du kill switch.
        """
        now = datetime.now()
        self.error_timestamps.append(now)
        
        # Compter les erreurs dans la fenêtre temporelle
        cutoff_time = now - timedelta(seconds=self.error_window_seconds)
        recent_errors = sum(1 for ts in self.error_timestamps if ts > cutoff_time)
        
        if recent_errors >= self.max_consecutive_errors:
            self.kill_switch_active = True
            self.kill_switch_triggers += 1
            logger.critical(
                f"🚨 KILL SWITCH ACTIVATED: {recent_errors} errors in {self.error_window_seconds}s "
                f"(threshold: {self.max_consecutive_errors})"
            )
    
    def record_success(self):
        """
        Enregistre un succès, réinitialise le compteur d'erreurs consécutives.
        """
        self.consecutive_errors = 0
        # Ne pas effacer error_timestamps pour garder l'historique récent
    
    def check_kill_switch(self) -> Tuple[bool, str]:
        """
        ✅ Kill Switch: Si 5 erreurs de suite → Arrêt d'Urgence
        
        Vérifie si le kill switch est activé.
        
        Returns:
            (safe: bool, reason: str)
        """
        if self.kill_switch_active:
            reason = (
                f"🚨 KILL SWITCH ACTIVE: Too many recent errors "
                f"({len(self.error_timestamps)} in last {self.error_window_seconds}s). "
                f"Trading paused for safety."
            )
            logger.critical(reason)
            return False, reason
        
        return True, "Kill switch inactive"
    
    def reset_kill_switch(self):
        """
        Réinitialise le kill switch manuellement (après investigation).
        """
        self.kill_switch_active = False
        self.error_timestamps.clear()
        self.consecutive_errors = 0
        logger.info("🔓 Kill switch manually reset")
    
    def full_safety_check(
        self,
        tick_timestamp: datetime,
        probabilities: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Effectue tous les checks de sécurité avant un trade.
        
        Args:
            tick_timestamp: Timestamp du tick de données
            probabilities: Probabilités de sortie du modèle
            
        Returns:
            (safe: bool, report: dict)
        """
        self.total_checks += 1
        
        report = {
            "timestamp": datetime.now(),
            "checks": {}
        }
        
        # 1. Check Kill Switch
        kill_switch_ok, kill_reason = self.check_kill_switch()
        report["checks"]["kill_switch"] = {
            "passed": kill_switch_ok,
            "reason": kill_reason
        }
        
        if not kill_switch_ok:
            return False, report
        
        # 2. Check Latency
        latency_ok, latency_reason, latency_ms = self.check_latency(tick_timestamp)
        report["checks"]["latency"] = {
            "passed": latency_ok,
            "reason": latency_reason,
            "latency_ms": latency_ms
        }
        
        if not latency_ok:
            return False, report
        
        # 3. Check Entropy
        entropy_ok, entropy_reason, entropy = self.check_entropy(probabilities)
        report["checks"]["entropy"] = {
            "passed": entropy_ok,
            "reason": entropy_reason,
            "entropy": entropy
        }
        
        if not entropy_ok:
            return False, report
        
        # All checks passed
        report["safe"] = True
        logger.info("✅ All safety checks passed")
        
        return True, report
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du garde de sécurité.
        """
        return {
            "total_checks": self.total_checks,
            "latency_rejects": self.latency_rejects,
            "entropy_rejects": self.entropy_rejects,
            "kill_switch_triggers": self.kill_switch_triggers,
            "kill_switch_active": self.kill_switch_active,
            "recent_errors": len(self.error_timestamps)
        }


class DynamicRiskManager:
    """
    ✅ Risk Management Dynamique basé sur la confiance du modèle.
    
    Le SL n'est pas fixe. Il est calculé via l'ATR actuel.
    Règle: Si Confiance Modèle > 0.8 → SL Large (ATR x 2). Sinon → SL Serré.
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.8,
        sl_atr_mult_confident: float = 2.0,
        sl_atr_mult_uncertain: float = 1.5,
        tp_atr_mult_confident: float = 3.0,
        tp_atr_mult_uncertain: float = 2.0
    ):
        """
        Initialise le gestionnaire de risque dynamique.
        
        Args:
            high_confidence_threshold: Seuil de confiance élevée (défaut: 0.8)
            sl_atr_mult_confident: Multiplicateur ATR pour SL en haute confiance (défaut: 2.0)
            sl_atr_mult_uncertain: Multiplicateur ATR pour SL en basse confiance (défaut: 1.5)
            tp_atr_mult_confident: Multiplicateur ATR pour TP en haute confiance (défaut: 3.0)
            tp_atr_mult_uncertain: Multiplicateur ATR pour TP en basse confiance (défaut: 2.0)
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.sl_atr_mult_confident = sl_atr_mult_confident
        self.sl_atr_mult_uncertain = sl_atr_mult_uncertain
        self.tp_atr_mult_confident = tp_atr_mult_confident
        self.tp_atr_mult_uncertain = tp_atr_mult_uncertain
    
    def calculate_sl_tp(
        self,
        signal: str,
        current_price: float,
        atr: float,
        model_confidence: float
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calcule le Stop Loss et Take Profit dynamiques basés sur la confiance du modèle.
        
        Args:
            signal: 'buy' ou 'sell'
            current_price: Prix actuel
            atr: Average True Range actuel
            model_confidence: Confiance du modèle (0.0 à 1.0)
            
        Returns:
            (sl: float, tp: float, info: dict)
        """
        # Déterminer si on est en haute confiance
        is_confident = model_confidence >= self.high_confidence_threshold
        
        # Sélectionner les multiplicateurs appropriés
        sl_mult = self.sl_atr_mult_confident if is_confident else self.sl_atr_mult_uncertain
        tp_mult = self.tp_atr_mult_confident if is_confident else self.tp_atr_mult_uncertain
        
        # Calculer SL et TP
        if signal == "buy":
            sl = current_price - (sl_mult * atr)
            tp = current_price + (tp_mult * atr)
        elif signal == "sell":
            sl = current_price + (sl_mult * atr)
            tp = current_price - (tp_mult * atr)
        else:
            raise ValueError(f"Invalid signal: {signal}")
        
        info = {
            "model_confidence": model_confidence,
            "is_confident": is_confident,
            "sl_multiplier": sl_mult,
            "tp_multiplier": tp_mult,
            "atr": atr,
            "risk_reward_ratio": tp_mult / sl_mult
        }
        
        logger.info(
            f"📊 Dynamic Risk: {'HIGH' if is_confident else 'LOW'} confidence "
            f"({model_confidence:.2f}) → SL={sl_mult}×ATR, TP={tp_mult}×ATR"
        )
        
        return sl, tp, info


# Fonctions utilitaires pour tests
def test_latency_guard():
    """Test unitaire pour la garde de latence."""
    guard = InferenceSafetyGuard(max_latency_ms=200.0)
    
    # Test 1: Latence acceptable
    recent_time = datetime.now() - timedelta(milliseconds=50)
    passed, reason, latency = guard.check_latency(recent_time)
    assert passed, "Should pass with low latency"
    assert latency < 200
    
    # Test 2: Latence excessive
    old_time = datetime.now() - timedelta(milliseconds=500)
    passed, reason, latency = guard.check_latency(old_time)
    assert not passed, "Should fail with high latency"
    assert latency > 200
    
    logger.info("✅ Latency guard tests passed")


def test_entropy_filter():
    """Test unitaire pour le filtre d'entropie."""
    guard = InferenceSafetyGuard(max_entropy=0.8)
    
    # Test 1: Haute confiance (faible entropie)
    probs_confident = np.array([0.9, 0.05, 0.05])
    passed, reason, entropy = guard.check_entropy(probs_confident)
    assert passed, "Should pass with low entropy"
    assert entropy < 0.8
    
    # Test 2: Modèle confus (haute entropie)
    probs_confused = np.array([0.33, 0.33, 0.34])  # Distribution quasi-uniforme
    passed, reason, entropy = guard.check_entropy(probs_confused)
    assert not passed, "Should fail with high entropy"
    assert entropy > 0.8
    
    logger.info("✅ Entropy filter tests passed")


def test_kill_switch():
    """Test unitaire pour le kill switch."""
    guard = InferenceSafetyGuard(max_consecutive_errors=3, error_window_seconds=60)
    
    # Simuler des erreurs
    for i in range(3):
        guard.record_error()
    
    passed, reason = guard.check_kill_switch()
    assert not passed, "Kill switch should be active"
    assert guard.kill_switch_active
    
    # Reset
    guard.reset_kill_switch()
    passed, reason = guard.check_kill_switch()
    assert passed, "Kill switch should be inactive after reset"
    
    logger.info("✅ Kill switch tests passed")


def test_dynamic_risk():
    """Test unitaire pour le risk management dynamique."""
    rm = DynamicRiskManager()
    
    # Test 1: Haute confiance
    sl, tp, info = rm.calculate_sl_tp("buy", 50000.0, 100.0, 0.85)
    assert info["is_confident"]
    assert info["sl_multiplier"] == 2.0
    assert info["tp_multiplier"] == 3.0
    
    # Test 2: Basse confiance
    sl, tp, info = rm.calculate_sl_tp("buy", 50000.0, 100.0, 0.65)
    assert not info["is_confident"]
    assert info["sl_multiplier"] == 1.5
    assert info["tp_multiplier"] == 2.0
    
    logger.info("✅ Dynamic risk management tests passed")


if __name__ == "__main__":
    # Run tests
    test_latency_guard()
    test_entropy_filter()
    test_kill_switch()
    test_dynamic_risk()
    logger.info("✅ All inference safety tests passed")
