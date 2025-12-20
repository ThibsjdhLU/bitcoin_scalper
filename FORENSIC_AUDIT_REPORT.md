# Forensic Code Audit Report

**Date:** 2025-12-20  
**Auditor:** GitHub Copilot  
**Scope:** Verify critical features are WIRED into the execution loop

---

## Methodology

Followed the execution flow:
1. **Entry Point:** `src/bitcoin_scalper/engine_main.py`
2. **Traced:** Imports to `TradingEngine` in `core/engine.py`
3. **Inspected:** The `process_tick` method (main loop handler)

---

## Audit Results Summary

| Component | File Path | Wired? | Evidence |
|:----------|:----------|:-------|:---------|
| **FracDiff Logic** | `utils/math_tools.py` | ✅ Yes (Exists) | Line 58: `def frac_diff_ffd(...)` |
| **FracDiff Usage** | `core/feature_engineering.py` | ❌ **UNWIRED** | Not imported, not called |
| **Paper Client** | `connectors/paper.py` | ✅ Yes (Exists) | Line 75: `class PaperMT5Client` |
| **Paper Logic** | `execute_order` method | ✅ **FUNCTIONAL** | Updates positions, balance, logs |
| **Deep Learning** | `models/deep_learning/lstm.py` | ❌ **DEAD CODE** | Never imported in engine |

---

## Detailed Findings

### 1. FracDiff (UNWIRED) 🔴

**The Math Exists but is NOT Connected to the Feature Pipeline**

- ✅ `utils/math_tools.py` line 58: `frac_diff_ffd()` is fully implemented
- ✅ `data/preprocessing.py` line 19: Imports and uses `frac_diff_ffd`
- ❌ `core/feature_engineering.py`: **NO import of `frac_diff_ffd`**
- ❌ `core/engine.py` → `process_tick()` → `feature_eng.add_indicators()` does NOT call `frac_diff_ffd`

**Evidence from `core/feature_engineering.py` imports (lines 1-9):**
```python
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, TSIIndicator, ...
# NO: from ..utils.math_tools import frac_diff_ffd
```

**Verdict:** FracDiff math is available but NOT used in the main trading engine feature pipeline.

---

### 2. Paper Trading (FUNCTIONAL) ✅

**Paper Mode is Properly Wired and Contains Real Logic**

- ✅ `engine_main.py` line 222: `paper_client = PaperMT5Client(...)`
- ✅ `engine_main.py` lines 238-255: TradingEngine initialized with `paper_client`
- ✅ `connectors/paper.py` line 288-355: `_execute_paper_order()` contains real logic

**Evidence from `connectors/paper.py`:**
```python
# Line 308: Apply slippage
fill_price = self._apply_slippage(current_price, action)

# Line 325: Add to positions
self.positions.append(position)

# Line 339: Record in history
self.order_history.append(order_record)

# Line 522: Balance updated on close
self.balance += position.profit
```

**Verdict:** Paper Trading is FUNCTIONAL with proper position tracking and P&L calculation.

---

### 3. Deep Learning (DEAD CODE) 🔴

**LSTM Model Exists but is Never Used**

- ✅ `models/deep_learning/lstm.py` line 32: `class LSTMModel(nn.Module)` exists
- ✅ `models/deep_learning/torch_wrapper.py` line 37: `TorchModelWrapper` exists
- ❌ `core/engine.py`: **NO import of LSTMModel or TorchModelWrapper**
- ❌ `load_ml_model()` method only loads CatBoost `.cbm` or joblib `.pkl` files

**Evidence from `core/engine.py` lines 256-264:**
```python
try:
    from catboost import CatBoostClassifier
    self.ml_model = CatBoostClassifier().load_model(f"{model_path}_model.cbm")
except Exception as e2:
    self.ml_model = joblib.load(f"{model_path}_model.pkl")
```

**No `torch.load()` or `.pth`/`.pt` file handling exists.**

**Verdict:** Deep Learning models are DEAD CODE - never wired into execution loop.

---

## Recommendations

| Component | Status | Action Required |
|:----------|:-------|:----------------|
| **FracDiff** | 🔴 UNWIRED | Import and call `frac_diff_ffd` in `FeatureEngineering.add_indicators()` |
| **Paper Trading** | ✅ FUNCTIONAL | No action needed |
| **Deep Learning** | 🔴 DEAD CODE | Add PyTorch loading in `engine.py` OR remove unused code |

---

## Conclusion

The trading engine has:
- **Working:** Paper trading simulation with proper balance/position tracking
- **Broken:** FracDiff is implemented but not integrated into feature engineering
- **Dead:** Deep learning models (LSTM, Transformer) exist but are never loaded or used

This audit was conducted by analyzing `.py` source files only, ignoring `.md` documentation as per instructions.
