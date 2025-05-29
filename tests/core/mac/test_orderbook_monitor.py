import numpy as np
from bitcoin_scalper.core.orderbook_monitor import OrderBookMonitor
import pytest

def make_orderbook():
    return {
        'bids': [
            {'price': 99, 'volume': 5},
            {'price': 98, 'volume': 3},
            {'price': 97, 'volume': 2}
        ],
        'asks': [
            {'price': 101, 'volume': 4},
            {'price': 102, 'volume': 2},
            {'price': 103, 'volume': 1}
        ]
    }

def test_compute_imbalance():
    ob = make_orderbook()
    mon = OrderBookMonitor(levels=3)
    imb = mon.compute_imbalance(ob)
    assert -1 <= imb <= 1

def test_detect_pressure():
    ob = make_orderbook()
    mon = OrderBookMonitor(levels=3)
    sig = mon.detect_pressure(ob, threshold=0.1)
    assert sig in [-1, 0, 1]

def test_get_best_bid_ask():
    ob = make_orderbook()
    mon = OrderBookMonitor()
    best = mon.get_best_bid_ask(ob)
    assert best['bid'] == 99
    assert best['ask'] == 101

def test_empty_orderbook():
    ob = {'bids': [], 'asks': []}
    mon = OrderBookMonitor()
    imb = mon.compute_imbalance(ob)
    assert imb == 0
    best = mon.get_best_bid_ask(ob)
    assert best['bid'] is None
    assert best['ask'] is None

def test_imbalance_missing_keys():
    ob = {}
    mon = OrderBookMonitor()
    imb = mon.compute_imbalance(ob)
    assert imb == 0

def test_imbalance_non_numeric():
    ob = {'bids': [{'price': 99, 'volume': 'foo'}], 'asks': [{'price': 101, 'volume': 4}]}
    mon = OrderBookMonitor()
    with pytest.raises(TypeError):
        mon.compute_imbalance(ob)

def test_detect_pressure_threshold():
    ob = make_orderbook()
    mon = OrderBookMonitor(levels=3)
    sig = mon.detect_pressure(ob, threshold=0)
    assert sig in [-1, 0, 1]

def test_get_best_bid_ask_empty():
    ob = {}
    mon = OrderBookMonitor()
    best = mon.get_best_bid_ask(ob)
    assert best['bid'] is None
    assert best['ask'] is None

def test_get_best_bid_ask_non_numeric():
    ob = {'bids': [{'price': 'foo', 'volume': 1}], 'asks': [{'price': 101, 'volume': 1}]}
    mon = OrderBookMonitor()
    best = mon.get_best_bid_ask(ob)
    assert best['bid'] == 'foo'
    assert best['ask'] == 101 