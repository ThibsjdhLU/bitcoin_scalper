import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from scipy import stats

class FractalAnalyzer:
    """
    Analyseur fractal avec indicateurs de Bill Williams et Elliott Wave
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.fractal_dimension = None
        self.bill_williams_fractals = None
        self.elliott_waves = None
        
    def calculate_fractal_dimension(self, data: pd.DataFrame) -> float:
        """
        Calcule la dimension fractale des données
        
        Args:
            data: DataFrame contenant les données de prix
            
        Returns:
            float: Dimension fractale
        """
        try:
            # Méthode de box-counting
            prices = data['close'].values
            n = len(prices)
            
            # Calcul des échelles
            scales = np.logspace(0, 3, 20)
            counts = []
            
            for scale in scales:
                # Discrétisation des prix
                boxes = np.floor(prices / scale)
                # Compte des boîtes uniques
                count = len(np.unique(boxes))
                counts.append(count)
                
            # Régression linéaire pour obtenir la dimension
            slope, _, _, _, _ = stats.linregress(np.log(scales), np.log(counts))
            self.fractal_dimension = -slope
            
            return self.fractal_dimension
            
        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {str(e)}")
            raise
            
    def detect_bill_williams_fractals(self, data: pd.DataFrame) -> Dict:
        """
        Détecte les fractales de Bill Williams
        
        Args:
            data: DataFrame contenant les données OHLC
            
        Returns:
            Dict: Fractales détectées
        """
        try:
            n = len(data)
            fractals = {
                'bullish': [],
                'bearish': []
            }
            
            for i in range(2, n-2):
                # Fractale haussière
                if (data['low'].iloc[i-2] > data['low'].iloc[i-1] and
                    data['low'].iloc[i-1] > data['low'].iloc[i] and
                    data['low'].iloc[i] < data['low'].iloc[i+1] and
                    data['low'].iloc[i+1] < data['low'].iloc[i+2]):
                    fractals['bullish'].append(i)
                    
                # Fractale baissière
                if (data['high'].iloc[i-2] < data['high'].iloc[i-1] and
                    data['high'].iloc[i-1] < data['high'].iloc[i] and
                    data['high'].iloc[i] > data['high'].iloc[i+1] and
                    data['high'].iloc[i+1] > data['high'].iloc[i+2]):
                    fractals['bearish'].append(i)
                    
            self.bill_williams_fractals = fractals
            return fractals
            
        except Exception as e:
            self.logger.error(f"Error detecting Bill Williams fractals: {str(e)}")
            raise
            
    def identify_elliott_waves(self, data: pd.DataFrame) -> Dict:
        """
        Identifie les vagues d'Elliott
        
        Args:
            data: DataFrame contenant les données de prix
            
        Returns:
            Dict: Vagues d'Elliott identifiées
        """
        try:
            n = len(data)
            waves = {
                'impulse': [],
                'corrective': []
            }
            
            # Détection des vagues impulsives
            for i in range(4, n-4):
                # Vague 1
                if self._is_impulse_wave(data, i, i+4):
                    waves['impulse'].append({
                        'start': i,
                        'end': i+4,
                        'type': '1'
                    })
                    
                # Vague 3
                if self._is_impulse_wave(data, i+4, i+8):
                    waves['impulse'].append({
                        'start': i+4,
                        'end': i+8,
                        'type': '3'
                    })
                    
                # Vague 5
                if self._is_impulse_wave(data, i+8, i+12):
                    waves['impulse'].append({
                        'start': i+8,
                        'end': i+12,
                        'type': '5'
                    })
                    
            # Détection des vagues correctives
            for i in range(2, n-2):
                if self._is_corrective_wave(data, i, i+2):
                    waves['corrective'].append({
                        'start': i,
                        'end': i+2,
                        'type': 'A'
                    })
                    
            self.elliott_waves = waves
            return waves
            
        except Exception as e:
            self.logger.error(f"Error identifying Elliott waves: {str(e)}")
            raise
            
    def _is_impulse_wave(self, data: pd.DataFrame, start: int, end: int) -> bool:
        """
        Vérifie si une séquence de prix forme une vague impulsive
        
        Args:
            data: DataFrame contenant les données
            start: Index de début
            end: Index de fin
            
        Returns:
            bool: True si c'est une vague impulsive
        """
        prices = data['close'].values[start:end]
        highs = data['high'].values[start:end]
        lows = data['low'].values[start:end]
        
        # Vérification des règles d'Elliott
        if len(prices) < 5:
            return False
            
        # Règle 1: La vague 2 ne peut pas retracer plus de 100% de la vague 1
        if lows[1] < prices[0]:
            return False
            
        # Règle 2: La vague 3 n'est jamais la plus courte
        if highs[2] - lows[2] < max(highs[0] - lows[0], highs[4] - lows[4]):
            return False
            
        # Règle 3: La vague 4 ne chevauche pas la vague 1
        if lows[3] < highs[0]:
            return False
            
        return True
        
    def _is_corrective_wave(self, data: pd.DataFrame, start: int, end: int) -> bool:
        """
        Vérifie si une séquence de prix forme une vague corrective
        
        Args:
            data: DataFrame contenant les données
            start: Index de début
            end: Index de fin
            
        Returns:
            bool: True si c'est une vague corrective
        """
        prices = data['close'].values[start:end]
        
        # Vagues correctives typiques (A-B-C)
        if len(prices) < 3:
            return False
            
        # Vérification du pattern A-B-C
        if (prices[1] < prices[0] and  # A
            prices[2] > prices[1] and  # B
            prices[2] < prices[0]):    # C
            return True
            
        return False
        
    def get_trading_signals(self) -> Dict:
        """
        Génère des signaux de trading basés sur l'analyse fractale
        
        Returns:
            Dict: Signaux de trading
        """
        try:
            signals = {
                'fractal_dimension': self.fractal_dimension,
                'bill_williams_signals': [],
                'elliott_wave_signals': []
            }
            
            # Signaux Bill Williams
            if self.bill_williams_fractals:
                for fractal_type, indices in self.bill_williams_fractals.items():
                    for idx in indices:
                        signals['bill_williams_signals'].append({
                            'type': fractal_type,
                            'index': idx,
                            'strength': self._calculate_signal_strength(idx)
                        })
                        
            # Signaux Elliott Wave
            if self.elliott_waves:
                for wave_type, waves in self.elliott_waves.items():
                    for wave in waves:
                        signals['elliott_wave_signals'].append({
                            'type': wave_type,
                            'wave': wave['type'],
                            'start': wave['start'],
                            'end': wave['end'],
                            'strength': self._calculate_wave_strength(wave)
                        })
                        
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            raise
            
    def _calculate_signal_strength(self, index: int) -> float:
        """
        Calcule la force d'un signal fractal
        
        Args:
            index: Index du signal
            
        Returns:
            float: Force du signal (0-1)
        """
        # Logique de calcul de la force du signal
        return 0.8  # Placeholder
        
    def _calculate_wave_strength(self, wave: Dict) -> float:
        """
        Calcule la force d'une vague d'Elliott
        
        Args:
            wave: Informations sur la vague
            
        Returns:
            float: Force de la vague (0-1)
        """
        # Logique de calcul de la force de la vague
        return 0.7  # Placeholder 