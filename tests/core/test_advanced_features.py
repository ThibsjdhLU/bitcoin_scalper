import unittest
import pandas as pd
import numpy as np
from bitcoin_scalper.core.feature_engineering import FeatureEngineering

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureEngineering()
        # Create a sample DataFrame with enough data for 200 periods + warmup
        self.n_rows = 400 # Increased to ensure dropna works if len > 300
        dates = pd.date_range(start='2023-01-01', periods=self.n_rows, freq='1min')
        self.df = pd.DataFrame({
            'close': np.random.uniform(20000, 21000, self.n_rows),
            'high': np.random.uniform(20000, 21000, self.n_rows),
            'low': np.random.uniform(20000, 21000, self.n_rows),
            'volume': np.random.uniform(1, 10, self.n_rows)
        }, index=dates)
        # Ensure high > low
        self.df['high'] = np.maximum(self.df['close'], self.df['high'])
        self.df['low'] = np.minimum(self.df['close'], self.df['low'])

    def test_multi_period_indicators(self):
        df_out = self.fe.add_indicators(self.df.copy())

        # Check RSI variants
        self.assertIn('rsi_7', df_out.columns)
        self.assertIn('rsi_14', df_out.columns)
        self.assertIn('rsi_21', df_out.columns)

        # Check ATR variants
        self.assertIn('atr_14', df_out.columns)
        self.assertIn('atr_21', df_out.columns)

        # Check BB variants
        self.assertIn('bb_width_20', df_out.columns)
        self.assertIn('bb_width_50', df_out.columns)

        # Check SMA/EMA variants
        self.assertIn('sma_50', df_out.columns)
        self.assertIn('ema_200', df_out.columns)

    def test_shift_logic(self):
        df_out = self.fe.add_indicators(self.df.copy())

        # RSI calculation (ta library) usually does not shift.
        # FeatureEngineering should shift it.
        # Let's verify shift manually.
        # Recalculate raw RSI 14
        from ta.momentum import RSIIndicator
        raw_rsi = RSIIndicator(self.df['close'], window=14, fillna=True).rsi()

        # The feature 'rsi_14' in df_out should be raw_rsi shifted by 1
        # Check alignment at index 50
        # If we dropped NaNs, indices align by label (datetime).
        # We need to match by index label.
        idx = df_out.index[50]
        # Previous index in original df
        prev_idx_loc = self.df.index.get_loc(idx) - 1
        prev_idx = self.df.index[prev_idx_loc]

        # Feature at T should equal Indicator at T-1
        val_feature = df_out.loc[idx, 'rsi_14']
        val_indicator = raw_rsi.loc[prev_idx]

        self.assertAlmostEqual(val_feature, val_indicator, places=5)

    def test_nan_handling(self):
        # We expect NaNs at the beginning due to shift, even if fillna=True was used in calculation,
        # unless dropna removed them.
        # Our code: if len > 300, dropna. Here len=400.
        # So we expect NO NaNs in the resulting DF if dropna worked well?
        # Wait, if we dropna THEN shift, we get 1 NaN at the start.
        # My code: dropna (removes warmup) THEN shift (adds 1 NaN).

        df_out = self.fe.add_indicators(self.df.copy())

        # First row should be NaN for shifted features
        self.assertTrue(np.isnan(df_out['rsi_14'].iloc[0]))

        # SMA 200 needs 200 points.
        # If dropna removed the first 200 points (approx), then we start clean.
        # Then shift adds 1 NaN.
        # So index 0 is NaN.
        # Index 1 should be valid.
        self.assertFalse(np.isnan(df_out['sma_200'].iloc[1]))

    def test_supertrend_manual(self):
        df_out = self.fe.add_indicators(self.df.copy())
        self.assertIn('supertrend', df_out.columns)
        self.assertIn('supertrend_direction', df_out.columns)
        # Check values are not all 0 or nan
        self.assertFalse(df_out['supertrend'].isna().all())
        self.assertFalse((df_out['supertrend'] == 0).all())

if __name__ == '__main__':
    unittest.main()
