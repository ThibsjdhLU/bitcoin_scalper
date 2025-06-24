import pytest
from bitcoin_scalper.core.data_cleaner import DataCleaner
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

def make_ticks():
    # Génère des ticks avec outliers et valeurs manquantes
    ticks = [
        {"bid": 1, "ask": 2, "volume": 0.1, "timestamp": "t1"},
        {"bid": 1.1, "ask": 2.1, "volume": 0.2, "timestamp": "t2"},
        {"bid": 100, "ask": 200, "volume": 10, "timestamp": "t3"},  # outlier
        {"bid": None, "ask": 2, "volume": 0.1, "timestamp": "t4"},   # missing
    ]
    return ticks

def make_ohlcv():
    ohlcv = [
        {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timestamp": "t1"},
        {"open": 1.1, "high": 2.1, "low": 0.6, "close": 1.6, "volume": 0.2, "timestamp": "t2"},
        {"open": 100, "high": 200, "low": 50, "close": 150, "volume": 10, "timestamp": "t3"},  # outlier
        {"open": None, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timestamp": "t4"}, # missing
    ]
    return ohlcv

def test_clean_ticks_outliers_missing():
    cleaner = DataCleaner(zscore_thresh=3.0)
    ticks = make_ticks()
    cleaned = cleaner.clean_ticks(ticks)
    # L'outlier et la ligne manquante doivent être supprimés
    assert all(t["bid"] < 10 for t in cleaned)
    assert all(t["bid"] is not None for t in cleaned)
    assert len(cleaned) == 2

def test_clean_ohlcv_outliers_missing():
    cleaner = DataCleaner()
    ohlcv = make_ohlcv()
    cleaned = cleaner.clean_ohlcv(ohlcv)
    assert all(o["open"] < 10 for o in cleaned)
    assert all(o["open"] is not None for o in cleaned)
    assert len(cleaned) == 2

def test_clean_ticks_isoforest():
    # Génère 50 ticks normaux + 2 outliers
    ticks = [{"bid": 1, "ask": 2, "volume": 0.1, "timestamp": str(i)} for i in range(50)]
    ticks += [{"bid": 100, "ask": 200, "volume": 10, "timestamp": "o1"}, {"bid": 120, "ask": 220, "volume": 12, "timestamp": "o2"}]
    cleaner = DataCleaner(anomaly_method="isoforest", contamination=0.05)
    cleaned = cleaner.clean_ticks(ticks)
    # Les outliers doivent être supprimés
    assert all(t["bid"] < 10 for t in cleaned)
    assert len(cleaned) < len(ticks)

def test_clean_ohlcv_isoforest():
    ohlcv = [{"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timestamp": str(i)} for i in range(50)]
    ohlcv += [{"open": 100, "high": 200, "low": 50, "close": 150, "volume": 10, "timestamp": "o1"}]
    cleaner = DataCleaner(anomaly_method="isoforest", contamination=0.05)
    cleaned = cleaner.clean_ohlcv(ohlcv)
    assert all(o["open"] < 10 for o in cleaned)
    assert len(cleaned) < len(ohlcv)

def test_clean_empty():
    cleaner = DataCleaner()
    assert cleaner.clean_ticks([]) == []
    assert cleaner.clean_ohlcv([]) == []

def test_clean_ticks_missing_columns():
    cleaner = DataCleaner()
    # Tick sans colonne 'ask' - doit lever une KeyError lors de la création du DataFrame ou dropna
    ticks = [{"bid": 1, "volume": 0.1, "timestamp": "t1"}]
    # Attendre une KeyError car la colonne 'ask' est manquante pour dropna(subset=...)
    with pytest.raises(KeyError, match=r"\['ask'\]"):
        cleaner.clean_ticks(ticks)

def test_clean_ohlcv_missing_columns():
    cleaner = DataCleaner()
    # OHLCV sans colonne 'close' - doit lever une KeyError lors de la création du DataFrame ou dropna
    ohlcv = [{"open": 1, "high": 2, "low": 0.5, "volume": 0.1, "timestamp": "t1"}]
    # Attendre une KeyError car la colonne 'close' est manquante pour dropna(subset=...)
    with pytest.raises(KeyError, match=r"\['close'\]"):
        cleaner.clean_ohlcv(ohlcv)

def test_clean_ticks_all_outliers():
    cleaner = DataCleaner(zscore_thresh=0.5) # Seuil plus strict pour forcer la détection
    # Créer des données où une valeur est clairement un outlier
    ticks = make_ticks()
    # Remplacer une valeur pour qu'elle soit un outlier flagrant
    ticks[2]["bid"] = 100000.0
    cleaned = cleaner.clean_ticks(ticks)
    # Vérifier qu'aucun des dictionnaires restants ne contient la valeur outlier
    assert all(t["bid"] != 100000.0 for t in cleaned)
    # Vérifier qu'il reste au moins un tick valide
    assert len(cleaned) >= 1

def test_clean_ohlcv_all_outliers():
    cleaner = DataCleaner(zscore_thresh=0.5) # Seuil plus strict
    ohlcv = make_ohlcv()
    # Remplacer une valeur pour qu'elle soit un outlier flagrant
    ohlcv[2]["close"] = -1000.0
    cleaned = cleaner.clean_ohlcv(ohlcv)
    # Vérifier qu'aucun des dictionnaires restants ne contient la valeur outlier
    assert all(o["close"] != -1000.0 for o in cleaned)
    # Vérifier qu'il reste au moins un OHLCV valide
    assert len(cleaned) >= 1

def test_clean_ticks_small_sample():
    cleaner = DataCleaner()
    ticks = [{"bid": 1, "ask": 2, "volume": 0.1, "timestamp": "t1"}]
    cleaned = cleaner.clean_ticks(ticks)
    assert len(cleaned) == 1

def test_clean_ohlcv_small_sample():
    cleaner = DataCleaner()
    ohlcv = [{"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timestamp": "t1"}]
    cleaned = cleaner.clean_ohlcv(ohlcv)
    assert len(cleaned) == 1

def test_clean_ticks_nan_values():
    cleaner = DataCleaner()
    ticks = [{"bid": np.nan, "ask": 2, "volume": 0.1, "timestamp": "t1"}]
    cleaned = cleaner.clean_ticks(ticks)
    assert cleaned == []

def test_clean_ohlcv_nan_values():
    cleaner = DataCleaner()
    ohlcv = [{"open": np.nan, "high": 2, "low": 0.5, "close": 1.5, "volume": 0.1, "timestamp": "t1"}]
    cleaned = cleaner.clean_ohlcv(ohlcv)
    assert cleaned == []

def test_clean_ticks_exception():
    cleaner = DataCleaner()
    ticks = make_ticks()
    # Mock une méthode de DataFrame appelée pendant le nettoyage (ex: dropna ou calcul zscore)
    with patch.object(pd.DataFrame, 'dropna', side_effect=Exception("Simulated DataFrame Error")):
        with pytest.raises(Exception, match="Simulated DataFrame Error"):
            cleaner.clean_ticks(ticks)

def test_clean_ohlcv_exception():
    cleaner = DataCleaner()
    ohlcv = make_ohlcv()
    # Mock une méthode de DataFrame appelée pendant le nettoyage
    with patch.object(pd.DataFrame, 'dropna', side_effect=Exception("Simulated DataFrame Error")):
        with pytest.raises(Exception, match="Simulated DataFrame Error"):
            cleaner.clean_ohlcv(ohlcv)

def test_correct_outliers_and_gaps(tmp_path):
    cleaner = DataCleaner()
    # Génère un DataFrame OHLCV avec outliers, trous, négatifs, incohérences
    idx = pd.date_range("2024-01-01 00:00", periods=10, freq="min").delete([3, 7])  # trous aux index 3 et 7
    df = pd.DataFrame({
        "open": [1, 1.1, 1.2, 10, 1.3, 1.4, -2, 1.5, 1.6, 1.7],
        "high": [2, 2.1, 2.2, 20, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
        "low": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
        "close": [1.5, 1.6, 1.7, 30, 1.8, 1.9, 1.95, 2.0, 2.1, 2.2],
        "volume": [0.1, 0.2, 0.3, 100, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }, index=idx)
    report_path = tmp_path / "correction_report.json"
    df_corr = cleaner.correct_outliers_and_gaps(df, freq="1min", outlier_zscore=3.0, winsorize=True, report_path=str(report_path))
    # Vérifie que le DataFrame est réindexé (plus de trous)
    assert len(df_corr) == 10
    # Vérifie que les valeurs négatives ont été corrigées
    assert (df_corr[["open", "high", "low", "close", "volume"]] >= 0).all().all()
    # Vérifie que le rapport JSON est généré
    assert report_path.exists()
    # Vérifie que les outliers ont été corrigés (plus de valeurs extrêmes)
    assert (df_corr[["open", "close", "volume"]] < 50).all().all()

def test_check_temporal_consistency(tmp_path):
    cleaner = DataCleaner()
    # Génère un DataFrame avec des incohérences
    idx = pd.date_range("2024-01-01 00:00", periods=10, freq="min").delete([3, 7])
    df = pd.DataFrame({
        "open": np.arange(10),
        "target_2m": [1]*8 + [np.nan, np.nan],
    }, index=idx)
    # Ajoute un doublon
    df_dup = pd.concat([df, df.iloc[[-1]]])
    df_dup = df_dup.sort_index()
    # Ajoute un timestamp dans le futur
    future_idx = pd.DatetimeIndex([pd.Timestamp.utcnow() + pd.Timedelta("1D")])
    df_dup.loc[future_idx, "open"] = 42
    report_path = tmp_path / "temporal_report.json"
    report = cleaner.check_temporal_consistency(df_dup, target_cols=["target_2m"], freq="1min", report_path=str(report_path))
    # Vérifie la détection des gaps, doublons, timestamps futurs, look-ahead
    assert "gaps" in report and len(report["gaps"]) > 0
    assert "duplicates" in report and len(report["duplicates"]) > 0
    assert "future_timestamps" in report and len(report["future_timestamps"]) > 0
    assert "lookahead" in report and "target_2m" in report["lookahead"]
    # Vérifie que le rapport JSON est généré
    assert report_path.exists()

def test_audit_data_quality(tmp_path):
    cleaner = DataCleaner()
    idx = pd.date_range("2024-01-01 00:00", periods=10, freq="min")
    df = pd.DataFrame({
        "open": np.arange(10),
        "high": np.arange(10)+1,
        "low": np.arange(10)-1,
        "close": np.arange(10),
        "volume": np.ones(10),
        "target_2m": [1]*8 + [np.nan, np.nan],
    }, index=idx)
    report = cleaner.audit_data_quality(df, label_cols=["target_2m"], report_dir=str(tmp_path), prefix="test_")
    # Vérifie que tous les rapports sont générés
    for key in ["clean_market", "correction", "temporal", "label_distribution"]:
        assert key in report
        assert (tmp_path / report[key].split("/")[-1]).exists()
    # Vérifie que le rapport global est généré
    assert (tmp_path / "test_global_audit.json").exists()

# TODO: Ajouter des tests pour les cas limites (listes vides, un seul élément)
# TODO: Ajouter des tests pour différents zscore_thresh
# TODO: Ajouter des tests pour la gestion des timestamps (conversion, format invalide)
# TODO: Ajouter des tests qui vérifient le contenu exact des données nettoyées (pas seulement la longueur) 