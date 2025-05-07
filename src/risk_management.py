def validate_trade_signal(self, signal: Dict, market_data: pd.DataFrame) -> bool:
    """
    Valide un signal de trading en fonction des critères de risque.
    
    Args:
        signal (Dict): Signal de trading à valider
        market_data (pd.DataFrame): Données de marché
        
    Returns:
        bool: True si le signal est valide, False sinon
    """
    try:
        print("\n=== Validation du signal de trading ===")
        
        # Calcul du risque de corrélation
        correlation_risk = self._calculate_correlation_risk(market_data)
        print(f"Risque de corrélation: {correlation_risk:.2f}")
        
        if correlation_risk > 0.7:  # Seuil assoupli de 0.8 à 0.7
            print("Signal rejeté: Risque de corrélation trop élevé")
            return False
            
        # Calcul de la fraction de Kelly
        kelly_fraction = self._calculate_kelly_fraction(signal, market_data)
        print(f"Fraction de Kelly: {kelly_fraction:.2f}")
        
        if kelly_fraction < 0.05:  # Seuil assoupli de 0.1 à 0.05
            print("Signal rejeté: Fraction de Kelly trop faible")
            return False
            
        print("Signal validé avec succès")
        return True
        
    except Exception as e:
        print(f"Erreur lors de la validation du signal: {str(e)}")
        return False

def _calculate_kelly_fraction(self, signal, market_data):
    """
    Calcule la fraction de Kelly optimale pour le position sizing.
    
    Args:
        signal (dict): Signal de trading
        market_data (pd.DataFrame): Données de marché
        
    Returns:
        float: Fraction de Kelly calculée
    """
    try:
        # Calcul du win rate historique
        win_rate = self._calculate_win_rate(market_data)
        
        # Calcul du ratio gain/perte
        profit_loss_ratio = self._calculate_profit_loss_ratio(market_data)
        
        # Calcul de la volatilité
        volatility = market_data['close'].pct_change().std()
        
        # Ajustement de la fraction de Kelly en fonction de la volatilité
        kelly_fraction = (win_rate * profit_loss_ratio - (1 - win_rate)) / profit_loss_ratio
        
        # Ajustement en fonction de la volatilité (réduction si volatilité élevée)
        if volatility > 0.02:  # Seuil de volatilité de 2%
            kelly_fraction *= 0.8
            
        # Ajustement en fonction de la force du signal
        kelly_fraction *= min(1.0, signal.get('strength', 0.5))
        
        # Limites de sécurité
        kelly_fraction = max(0.01, min(0.5, kelly_fraction))
        
        return kelly_fraction
        
    except Exception as e:
        print(f"Erreur lors du calcul de la fraction de Kelly: {str(e)}")
        return 0.05  # Valeur par défaut conservative 

def _calculate_win_rate(self, market_data):
    """
    Calcule le taux de réussite historique des trades.
    
    Args:
        market_data (pd.DataFrame): Données de marché
        
    Returns:
        float: Taux de réussite (entre 0 et 1)
    """
    try:
        # Calcul des rendements
        returns = market_data['close'].pct_change()
        
        # Identification des trades gagnants et perdants
        winning_trades = len(returns[returns > 0])
        total_trades = len(returns[returns != 0])
        
        if total_trades == 0:
            return 0.5  # Valeur par défaut si pas assez de données
            
        return winning_trades / total_trades
        
    except Exception as e:
        print(f"Erreur lors du calcul du win rate: {str(e)}")
        return 0.5

def _calculate_profit_loss_ratio(self, market_data):
    """
    Calcule le ratio gain/perte moyen.
    
    Args:
        market_data (pd.DataFrame): Données de marché
        
    Returns:
        float: Ratio gain/perte
    """
    try:
        # Calcul des rendements
        returns = market_data['close'].pct_change()
        
        # Calcul des gains et pertes moyens
        avg_gain = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        
        if avg_loss == 0:
            return 2.0  # Valeur par défaut si pas de pertes
            
        return avg_gain / avg_loss
        
    except Exception as e:
        print(f"Erreur lors du calcul du ratio gain/perte: {str(e)}")
        return 2.0 