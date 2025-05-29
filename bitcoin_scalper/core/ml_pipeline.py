import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import shap
# LIME import à ajouter si besoin
import os
from bitcoin_scalper.core.dvc_manager import DVCManager
import logging

class EarlyStopping:
    """
    Callback d'arrêt anticipé pour l'entraînement des modèles PyTorch.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class DNNClassifier(nn.Module):
    """
    Réseau de neurones fully-connected pour la classification.
    """
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class CNN1DClassifier(nn.Module):
    """
    Réseau de neurones convolutionnel 1D pour la classification séquentielle.
    """
    def __init__(self, input_dim, num_filters=16, kernel_size=3, output_dim=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, num_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, output_dim)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class LSTMClassifier(nn.Module):
    """
    Réseau LSTM pour la classification séquentielle.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class TransformerClassifier(nn.Module):
    """
    Classifieur basé sur Transformer pour séquences temporelles.
    """
    def __init__(self, input_dim, nhead=2, num_layers=1, hidden_dim=32, output_dim=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = x.permute(1, 0, 2)
        out = self.transformer(x)
        out = out[-1, :, :]
        return self.fc(out)

class MLPipeline:
    """
    Pipeline ML avancée pour la classification/prédiction de signaux de trading.
    Supporte RandomForest, XGBoost, DNN, LSTM, Transformer, CNN1D.
    Gère split, CV, tuning, prédiction, sauvegarde/chargement (DVC ready), explicabilité, callbacks, monitoring.
    """
    def __init__(self, model_type: str = "random_forest", params: Optional[Dict[str, Any]] = None,
                 device: str = "cpu", dvc_track: bool = False, random_state: Optional[int] = None,
                 callbacks: Optional[List[Callable]] = None):
        self.model_type = model_type
        self.params = params or {}
        self.device = torch.device(device)
        self.dvc_track = dvc_track
        self.random_state = random_state
        self.callbacks = callbacks or []
        self.model = None  # Initialisé dans fit pour PyTorch
        self.metrics = {}
        self.dvc = DVCManager() if dvc_track else None
        self.logger = logging.getLogger("MLPipeline")
        self._set_seeds()
        supported_types = ["random_forest", "xgboost", "dnn", "lstm", "transformer", "cnn1d"]
        if self.model_type not in supported_types:
            raise ValueError(f"Modèle non supporté: {self.model_type}")
        if self.model_type in ["random_forest", "xgboost"]:
            self.model = self._init_model()

    def _set_seeds(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _init_model(self, input_dim: Optional[int] = None):
        if self.model_type == "random_forest":
            return RandomForestClassifier(random_state=self.random_state, **self.params)
        elif self.model_type == "xgboost":
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=self.random_state, **self.params)
        elif self.model_type == "dnn":
            return DNNClassifier(input_dim, self.params.get("hidden_dim", 32), self.params.get("output_dim", 2))
        elif self.model_type == "cnn1d":
            return CNN1DClassifier(input_dim, self.params.get("num_filters", 16), self.params.get("kernel_size", 3), self.params.get("output_dim", 2))
        elif self.model_type == "lstm":
            return LSTMClassifier(input_dim, self.params.get("hidden_dim", 32), self.params.get("num_layers", 1), self.params.get("output_dim", 2))
        elif self.model_type == "transformer":
            return TransformerClassifier(input_dim, self.params.get("nhead", 2), self.params.get("num_layers", 1), self.params.get("hidden_dim", 32), self.params.get("output_dim", 2))
        else:
            raise ValueError(f"Modèle non supporté: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series, val_split: float = 0.2, cv: Optional[int] = None, epochs: int = 10, batch_size: int = 32, early_stopping: Optional[EarlyStopping] = None):
        if self.model_type in ["random_forest", "xgboost"]:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, shuffle=False, random_state=self.random_state)
            if cv:
                scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="accuracy")
                self.metrics["cv_accuracy"] = np.mean(scores)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            self.metrics["val_accuracy"] = accuracy_score(y_val, y_pred)
            self.metrics["val_f1"] = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            try:
                val_proba = self.model.predict_proba(X_val)
                if val_proba is not None and val_proba.shape[1] > 2:
                    self.metrics["val_auc"] = roc_auc_score(y_val, val_proba, multi_class='ovr', average='weighted')
                elif val_proba is not None and val_proba.shape[1] == 2:
                    self.metrics["val_auc"] = roc_auc_score(y_val, val_proba[:, 1])
                else:
                    self.metrics["val_auc"] = None
            except Exception:
                self.metrics["val_auc"] = None
            return self.metrics
        elif self.model_type in ["dnn", "lstm", "transformer", "cnn1d"]:
            X_np = np.asarray(X)
            y_np = np.asarray(y)
            input_dim = X_np.shape[-1]
            if self.model is None or not hasattr(self.model, 'net'):
                self.model = self._init_model(input_dim)
            n = len(X_np)
            split = int(n * (1 - val_split))
            X_train, X_val = X_np[:split], X_np[split:]
            y_train, y_val = y_np[:split], y_np[split:]
            if self.model_type in ["lstm", "transformer", "cnn1d"]:
                tscv = TimeSeriesSplit(n_splits=5)
                # Pour simplifier, on garde le dernier split pour validation
                for train_idx, val_idx in tscv.split(X_np):
                    X_train, X_val = X_np[train_idx], X_np[val_idx]
                    y_train, y_val = y_np[train_idx], y_np[val_idx]
            model = self.model.to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.params.get("lr", 1e-3))
            best_val_loss = float('inf')
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                for i in range(0, len(X_train), batch_size):
                    xb = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32, device=self.device)
                    yb = torch.tensor(y_train[i:i+batch_size], dtype=torch.long, device=self.device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                    yv = torch.tensor(y_val, dtype=torch.long, device=self.device)
                    logits = model(Xv)
                    val_loss = criterion(logits, yv).item()
                    avg_val_loss = val_loss
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    self.metrics["val_accuracy"] = (preds == y_val).mean()
                    self.metrics["val_f1"] = f1_score(y_val, preds, average='weighted', zero_division=0)
                    try:
                        val_probs = torch.softmax(logits, dim=1).cpu().numpy()
                        if val_probs.shape[1] > 2:
                            self.metrics["val_auc"] = roc_auc_score(y_val, val_probs, multi_class='ovr', average='weighted')
                        elif val_probs.shape[1] == 2:
                            self.metrics["val_auc"] = roc_auc_score(y_val, val_probs[:, 1])
                        else:
                            self.metrics["val_auc"] = None
                    except Exception:
                        self.metrics["val_auc"] = None
                if early_stopping is not None and early_stopping(avg_val_loss):
                    self.logger.info(f"Early stopping à l'époque {epoch}")
                    break
                for cb in self.callbacks:
                    cb(epoch, model, self.metrics)
            return self.metrics
        else:
            raise ValueError(f"Modèle non supporté: {self.model_type}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_type in ["random_forest", "xgboost"]:
            return self.model.predict(X)
        elif self.model_type in ["dnn", "lstm", "transformer", "cnn1d"]:
            self.model.eval()
            X_np = np.asarray(X)
            with torch.no_grad():
                X_tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device)
                logits = self.model(X_tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            return preds
        else:
            raise ValueError(f"Modèle non supporté: {self.model_type}")

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.model_type in ["random_forest", "xgboost"]:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            else:
                return None
        elif self.model_type in ["dnn", "lstm", "transformer", "cnn1d"]:
            self.model.eval()
            X_np = np.asarray(X)
            with torch.no_grad():
                X_tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device)
                logits = self.model(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs
        else:
            return None

    def _pytorch_tuning(self, X, y, param_grid, cv=3, n_trials=20):
        import optuna
        def objective(trial):
            params = {k: trial.suggest_categorical(k, v) for k, v in param_grid.items()}
            input_dim = X.shape[1]
            model = self._init_model(input_dim)
            model = model.to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
            n = len(X)
            split = int(n * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            for epoch in range(10):
                model.train()
                for i in range(0, len(X_train), 32):
                    xb = torch.tensor(X_train[i:i+32], dtype=torch.float32, device=self.device)
                    yb = torch.tensor(y_train[i:i+32], dtype=torch.long, device=self.device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                yv = torch.tensor(y_val, dtype=torch.long, device=self.device)
                logits = model(Xv)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                acc = (preds == y_val).mean()
            return acc
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        self.model = self._init_model(X.shape[1])
        self.model = self.model.to(self.device)
        self.params.update(best_params)
        self.metrics["best_params"] = best_params
        self.metrics["best_score"] = study.best_value
        return self.metrics

    def tune(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, List[Any]], cv: int = 3, use_optuna: bool = False, n_trials: int = 20):
        if self.model_type in ["dnn", "lstm", "transformer", "cnn1d"]:
            return self._pytorch_tuning(np.asarray(X), np.asarray(y), param_grid, cv=cv, n_trials=n_trials)
        if use_optuna:
            import optuna
            def objective(trial):
                params = {k: trial.suggest_categorical(k, v) for k, v in param_grid.items()}
                model = self._init_model()
                model.set_params(**params)
                scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                return np.mean(scores)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            self.model.set_params(**best_params)
            self.metrics["best_params"] = best_params
            self.metrics["best_score"] = study.best_value
        else:
            grid = GridSearchCV(self.model, param_grid, cv=cv, scoring="accuracy")
            grid.fit(X, y)
            self.model = grid.best_estimator_
            self.metrics["best_params"] = grid.best_params_
            self.metrics["best_score"] = grid.best_score_
        return self.metrics

    def save(self, path: str):
        if self.model_type in ["dnn", "lstm", "transformer", "cnn1d"]:
            torch.save(self.model.state_dict(), path)
        else:
            joblib.dump(self.model, path)
        if self.dvc_track:
            self.dvc.add(path)
            self.dvc.commit(path)
            self.dvc.push()

    def load(self, path: str, input_dim: Optional[int] = None):
        if self.model_type in ["dnn", "lstm", "transformer", "cnn1d"]:
            if self.model is None:
                if input_dim is None:
                    raise ValueError("input_dim doit être fourni pour charger un modèle PyTorch.")
                self.model = self._init_model(input_dim)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model = self.model.to(self.device)
        else:
            self.model = joblib.load(path)
        if self.dvc_track:
            self.dvc.pull()

    def explain(self, X: pd.DataFrame, method: str = "shap", nsamples: int = 100):
        """Explicabilité du modèle (SHAP pour tabulaire, DeepExplainer pour PyTorch)."""
        if method == "shap" and self.model_type in ["random_forest", "xgboost"]:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            return shap_values
        elif method == "shap" and self.model_type in ["dnn", "lstm", "transformer", "cnn1d"]:
            self.model.eval()
            X_np = np.asarray(X)
            X_tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            explainer = shap.DeepExplainer(self.model, X_tensor[:nsamples])
            shap_values = explainer.shap_values(X_tensor[:nsamples])
            return shap_values
        else:
            raise NotImplementedError("Explicabilité non supportée pour ce modèle ou cette méthode.") 