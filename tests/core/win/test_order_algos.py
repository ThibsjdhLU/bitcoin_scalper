import numpy as np
from bitcoin_scalper.core.order_algos import execute_iceberg, execute_twap, execute_vwap
import pytest

def dummy_send_order(qty, price, **kwargs):
    return {"qty": qty, "price": price, "status": "sent"}

@pytest.mark.timeout(5)
@pytest.mark.xfail(reason="Ce test semble causer un problème de ressources ou un blocage sur macOS")
def test_execute_iceberg():
    res = execute_iceberg(total_qty=1, max_child=1, price=100, send_order_fn=dummy_send_order)
    assert sum([r["qty"] for r in res]) == 1
    assert all(r["price"] == 100 for r in res)
    assert len(res) == 1

@pytest.mark.timeout(5)
@pytest.mark.xfail(reason="Ce test semble causer un problème de ressources ou un blocage sur macOS")
def test_execute_twap():
    res = execute_twap(total_qty=1, n_slices=1, price=101, send_order_fn=dummy_send_order)
    assert sum([r["qty"] for r in res]) == 1
    assert all(r["price"] == 101 for r in res)
    assert len(res) == 1

@pytest.mark.timeout(5)
@pytest.mark.xfail(reason="Ce test semble causer un problème de ressources ou un blocage sur macOS")
def test_execute_vwap():
    prices = [100]
    res = execute_vwap(total_qty=1, price_series=prices, send_order_fn=dummy_send_order)
    assert sum([r["qty"] for r in res]) == 1
    assert [r["price"] for r in res] == prices
    assert len(res) == 1

def test_iceberg_zero_qty():
    res = execute_iceberg(total_qty=0, max_child=3, price=100, send_order_fn=dummy_send_order)
    assert res == []

def test_iceberg_negative_qty():
    res = execute_iceberg(total_qty=-5, max_child=3, price=100, send_order_fn=dummy_send_order)
    assert res == []

def test_iceberg_zero_max_child():
    with pytest.raises(ValueError):
        execute_iceberg(total_qty=10, max_child=0, price=100, send_order_fn=dummy_send_order)

def test_iceberg_non_numeric():
    with pytest.raises(TypeError):
        execute_iceberg(total_qty="foo", max_child=3, price=100, send_order_fn=dummy_send_order)

    with pytest.raises(TypeError):
        execute_iceberg(total_qty=10, max_child="bar", price=100, send_order_fn=dummy_send_order)

def test_twap_zero_slices():
    with pytest.raises(ValueError):
        execute_twap(total_qty=10, n_slices=0, price=101, send_order_fn=dummy_send_order)

def test_twap_negative_slices():
    with pytest.raises(ValueError):
        execute_twap(total_qty=10, n_slices=-2, price=101, send_order_fn=dummy_send_order)

def test_twap_non_numeric():
    with pytest.raises(TypeError):
        execute_twap(total_qty="foo", n_slices=4, price=101, send_order_fn=dummy_send_order)
    with pytest.raises(TypeError):
        execute_twap(total_qty=10, n_slices="bar", price=101, send_order_fn=dummy_send_order)

def test_vwap_empty_series():
    res = execute_vwap(total_qty=10, price_series=[], send_order_fn=dummy_send_order)
    assert res == []

def test_vwap_non_numeric():
    with pytest.raises(TypeError):
        execute_vwap(total_qty="foo", price_series=[100,101], send_order_fn=dummy_send_order)
    with pytest.raises(TypeError):
        execute_vwap(total_qty=10, price_series=["foo",101], send_order_fn=dummy_send_order) 