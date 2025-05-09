"""
Test des constantes disponibles dans MetaTrader5.
"""
import MetaTrader5 as mt5
import pytest


def test_mt5_constants():
    """Vérifie les constantes disponibles dans MT5."""
    # Initialiser MT5
    if not mt5.initialize():
        pytest.skip("MetaTrader5 non initialisé")

    try:
        # Vérifier les constantes d'ordre
        assert hasattr(mt5, "ORDER_TYPE_BUY")
        assert hasattr(mt5, "ORDER_TYPE_SELL")
        assert hasattr(mt5, "ORDER_TYPE_BUY_LIMIT")
        assert hasattr(mt5, "ORDER_TYPE_SELL_LIMIT")
        assert hasattr(mt5, "ORDER_TYPE_BUY_STOP")
        assert hasattr(mt5, "ORDER_TYPE_SELL_STOP")

        # Afficher les valeurs
        print("\nConstantes MT5 disponibles:")
        print(f"ORDER_TYPE_BUY = {mt5.ORDER_TYPE_BUY}")
        print(f"ORDER_TYPE_SELL = {mt5.ORDER_TYPE_SELL}")
        print(f"ORDER_TYPE_BUY_LIMIT = {mt5.ORDER_TYPE_BUY_LIMIT}")
        print(f"ORDER_TYPE_SELL_LIMIT = {mt5.ORDER_TYPE_SELL_LIMIT}")
        print(f"ORDER_TYPE_BUY_STOP = {mt5.ORDER_TYPE_BUY_STOP}")
        print(f"ORDER_TYPE_SELL_STOP = {mt5.ORDER_TYPE_SELL_STOP}")

    finally:
        mt5.shutdown()
