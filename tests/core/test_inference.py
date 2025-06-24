import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from bitcoin_scalper.core.inference import inference
from bitcoin_scalper.core.export import save_objects
from catboost import CatBoostClassifier

def make_df_minute():
    idx = pd.date_range('2024-06-01', periods=20, freq='T', tz='UTC')
    df = pd.DataFrame({
        '<OPEN>': np.linspace(70000, 70020, 20),
        '<HIGH>': np.linspace(70010, 70030, 20),
        '<LOW>': np.linspace(69990, 70010, 20),
        '<CLOSE>': np.linspace(70005, 70025, 20),
        '<TICKVOL>': np.random.randint(90, 130, 20).astype(np.float32),
    }, index=idx)
    return df

def make_and_save_model(tmp_path):
    df = make_df_minute()
    X = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']].copy()
    y = np.random.choice([-1, 0, 1], len(X))
    model = CatBoostClassifier(loss_function='MultiClass', iterations=5, verbose=0)
    model.fit(X, y)
    prefix = str(tmp_path / "test_infer")
    save_objects(model, None, None, None, prefix)
    return prefix, df

def test_inference_ok(tmp_path):
    prefix, df = make_and_save_model(tmp_path)
    sig = inference(df, path_prefix=prefix)
    assert isinstance(sig, pd.Series)
    assert sig.shape[0] == df.shape[0]
    assert set(sig.unique()) <= {-1, 0, 1}

def test_inference_missing_file(tmp_path):
    df = make_df_minute()
    with pytest.raises(FileNotFoundError):
        inference(df, path_prefix=str(tmp_path / "missing"))

def test_inference_bad_format(tmp_path):
    prefix, _ = make_and_save_model(tmp_path)
    # DataFrame sans colonnes requises
    df = pd.DataFrame({'foo': [1,2,3]})
    with pytest.raises(Exception):
        inference(df, path_prefix=prefix) 