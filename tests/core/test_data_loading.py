import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from bitcoin_scalper.core.data_loading import load_minute_csv, InvalidFrequencyError

def make_csv(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path

def test_load_minute_csv_ok():
    csv = """<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>\n2024.06.01,00:00,70000,70100,69900,70050,100,1,2\n2024.06.01,00:01,70050,70200,70000,70150,110,1,2\n2024.06.01,00:02,70150,70300,70100,70250,120,1,2\n"""
    path = make_csv(csv)
    df = load_minute_csv(path)
    assert list(df.columns) == ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
    assert df.index.tz.zone == 'UTC'
    assert df.shape == (3, 5)
    assert df['<OPEN>'].dtype == np.float32
    os.remove(path)

def test_load_minute_csv_doublons_nulls():
    csv = """<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>\n2024.06.01,00:00,70000,70100,69900,70050,100,1,2\n2024.06.01,00:00,70000,70100,69900,70050,100,1,2\n2024.06.01,00:01,70050,70200,70000,70150,,1,2\n2024.06.01,00:02,70150,70300,70100,70250,120,1,2\n"""
    path = make_csv(csv)
    df = load_minute_csv(path)
    # 1 doublon et 1 valeur nulle supprimés
    assert df.shape == (2, 5)
    os.remove(path)

def test_load_minute_csv_freq_error():
    csv = """<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>\n2024.06.01,00:00,70000,70100,69900,70050,100,1,2\n2024.06.01,00:02,70150,70300,70100,70250,120,1,2\n"""
    path = make_csv(csv)
    with pytest.raises(InvalidFrequencyError):
        load_minute_csv(path)
    os.remove(path)

def test_load_minute_csv_missing_cols():
    csv = """<DATE>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>\n2024.06.01,70000,70100,69900,70050,100,1,2\n"""
    path = make_csv(csv)
    with pytest.raises(ValueError):
        load_minute_csv(path)
    os.remove(path)

def test_load_minute_csv_types():
    csv = """<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>\n2024.06.01,00:00,70000,70100,69900,70050,100,1,2\n2024.06.01,00:01,70050,70200,70000,70150,110,1,2\n"""
    path = make_csv(csv)
    df = load_minute_csv(path)
    assert all(df[c].dtype == np.float32 for c in df.columns)
    os.remove(path) 