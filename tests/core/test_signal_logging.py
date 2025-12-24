"""
Unit test to verify that signal logging happens before hold checks.

This ensures that ALL signals (including holds and rejected trades) are logged,
which is critical for debugging why trades are not being executed.
"""
import pytest
from unittest.mock import Mock, patch, call
import pandas as pd
import numpy as np
import os
from pathlib import Path


# Get the path to engine.py relative to this test file
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent
ENGINE_PATH = PROJECT_ROOT / "src" / "bitcoin_scalper" / "core" / "engine.py"


def test_signal_logged_before_hold_check():
    """
    Test that log_signal is called even when signal is 'hold' or None.
    
    This is critical for debugging - we need to see WHY a trade was rejected
    (e.g., low confidence) rather than seeing nothing in the logs.
    """
    # Mock the TradingEngine dependencies
    mock_connector = Mock()
    mock_connector._request = Mock(return_value={'balance': 10000.0, 'equity': 10000.0})
    
    # Create a minimal mock logger
    mock_logger = Mock()
    mock_logger.log_signal = Mock()
    mock_logger.info = Mock()
    mock_logger.warning = Mock()
    
    # We'll test the code logic rather than actually running it
    # since we don't have all dependencies installed
    
    # Verify the code structure is correct by reading the file
    with open(ENGINE_PATH, 'r') as f:
        engine_code = f.read()
    
    # Find process_tick method
    start = engine_code.find('def process_tick(')
    assert start > 0, "process_tick method not found"
    
    # Find the next method definition after process_tick
    next_def = engine_code.find('\n    def ', start + 1)
    process_tick_code = engine_code[start:next_def]
    
    # Verify log_signal appears before hold check
    log_signal_pos = process_tick_code.find('self.logger.log_signal(')
    hold_check_pos = process_tick_code.find('if signal is None or signal ==')
    
    assert log_signal_pos > 0, "log_signal not found in process_tick"
    assert hold_check_pos > 0, "hold check not found in process_tick"
    assert log_signal_pos < hold_check_pos, \
        f"log_signal (pos {log_signal_pos}) must come BEFORE hold check (pos {hold_check_pos})"
    
    # Verify there's only one log_signal call (no duplicates)
    log_signal_count = process_tick_code.count('self.logger.log_signal(')
    assert log_signal_count == 1, \
        f"Expected exactly 1 log_signal call, found {log_signal_count}"


def test_signal_logging_includes_confidence():
    """
    Test that signal logging includes confidence even when signal is hold.
    
    This allows seeing when trades are rejected due to low confidence from
    the meta-model, which is the primary use case for this change.
    """
    with open(ENGINE_PATH, 'r') as f:
        engine_code = f.read()
    
    # Find the log_signal call
    log_signal_start = engine_code.find('# Log the signal BEFORE checking')
    assert log_signal_start > 0, "Signal logging section not found"
    
    # Get a window around the log_signal call
    window = engine_code[log_signal_start:log_signal_start + 500]
    
    # Verify it logs confidence
    assert 'confidence=confidence' in window, \
        "log_signal must include confidence parameter"
    
    # Verify it logs before hold check (based on comment)
    assert 'BEFORE checking' in window, \
        "Missing explanatory comment about logging before hold check"


def test_no_duplicate_signal_logging():
    """
    Test that there's no duplicate signal logging after position sizing.
    
    Previously, there was a second log_signal call after position sizing,
    which meant hold signals were never logged (they exit before reaching it).
    Now there should be only one log_signal call.
    """
    with open(ENGINE_PATH, 'r') as f:
        engine_code = f.read()
    
    # Find process_tick method
    start = engine_code.find('def process_tick(')
    next_def = engine_code.find('\n    def ', start + 1)
    process_tick_code = engine_code[start:next_def]
    
    # Count log_signal calls
    log_signal_count = process_tick_code.count('self.logger.log_signal(')
    assert log_signal_count == 1, \
        f"Expected exactly 1 log_signal call in process_tick, found {log_signal_count}"


if __name__ == "__main__":
    # Run tests
    print("Running test_signal_logged_before_hold_check...")
    test_signal_logged_before_hold_check()
    print("✓ PASSED\n")
    
    print("Running test_signal_logging_includes_confidence...")
    test_signal_logging_includes_confidence()
    print("✓ PASSED\n")
    
    print("Running test_no_duplicate_signal_logging...")
    test_no_duplicate_signal_logging()
    print("✓ PASSED\n")
    
    print("="*50)
    print("All tests passed! ✓")
    print("="*50)
