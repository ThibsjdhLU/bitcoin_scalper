import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
from bitcoin_scalper.core.data_loading import load_minute_csv
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.labeling import generate_labels
from bitcoin_scalper.core.balancing import balance_by_block
from bitcoin_scalper.core.splitting import split_dataset
from bitcoin_scalper.core.modeling import train_model, predict
from bitcoin_scalper.core.evaluation import evaluate_classification, evaluate_financial, plot_pnl_curve
from bitcoin_scalper.core.export import save_objects, load_objects
from bitcoin_scalper.core.inference import inference

# Logging global
logger = logging.getLogger("bitcoin_scalper.orchestrator")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet ML pour trading BTC minute : data_loading → feature_engineering → labeling → balancing → splitting → modeling → evaluation → export → inference"
    )
    parser.add_argument('--csv', required=True, help='Chemin du CSV minute brut')
    parser.add_argument('--model_prefix', default='model', help='Préfixe pour la sauvegarde des objets')
    parser.add_argument('--label_horizon', type=int, default=15, help='Horizon de prédiction pour le label (min)')
    parser.add_argument('--label_k', type=float, default=0.5, help='Multiplicateur du seuil dynamique pour le label')
    parser.add_argument('--block_duration', default='1D', help='Durée des blocs pour l\'équilibrage')
    parser.add_argument('--min_block_size', type=int, default=100, help='Taille minimale d\'un bloc pour l\'équilibrage')
    parser.add_argument('--split_method', default='fixed', choices=['fixed', 'purged_kfold'], help='Stratégie de split')
    parser.add_argument('--train_frac', type=float, default=0.7, help='Fraction train')
    parser.add_argument('--val_frac', type=float, default=0.15, help='Fraction validation')
    parser.add_argument('--test_frac', type=float, default=0.15, help='Fraction test')
    parser.add_argument('--purge_window', type=int, default=None, help='Fenêtre de purge pour purged_kfold')
    parser.add_argument('--tuning', default='optuna', choices=['optuna', 'grid'], help='Méthode de tuning LightGBM')
    parser.add_argument('--early_stopping_rounds', type=int, default=20, help='Early stopping rounds')
    parser.add_argument('--export', action='store_true', help='Sauvegarder les objets finaux')
    parser.add_argument('--inference', action='store_true', help='Effectuer l\'inférence sur le test set')
    parser.add_argument('--plot', action='store_true', help='Tracer la courbe de PnL')
    args = parser.parse_args()

    try:
        # 1. Chargement et nettoyage des données 1 minute
        logger.info('Chargement et nettoyage du CSV minute...')
        df_1min = load_minute_csv(args.csv, report_missing="rapport_trous_temporals.txt")
        df_1min.to_pickle('artf_data_cleaned_1min.pkl')

        # Agrégation des données en 5 minutes
        logger.info('Agrégation des données en 5 minutes...')
        df_5min = df_1min.resample('5min').agg({
            '<OPEN>': 'first',
            '<HIGH>': 'max',
            '<LOW>': 'min',
            '<CLOSE>': 'last',
            '<TICKVOL>': 'sum'
        }).dropna()
        df_5min.to_pickle('artf_data_cleaned_5min.pkl')

        # Stocker les DataFrames par timeframe dans un dictionnaire
        dfs = {
            "1min": df_1min,
            "5min": df_5min
        }

        # 2. Feature engineering multi-timeframe
        logger.info('Feature engineering multi-timeframe...')
        fe = FeatureEngineering()
        df_feat = fe.multi_timeframe(dfs)
        df_feat.to_pickle('artf_features.pkl')

        # 3. Labeling (sur les données 1min, après le join)
        logger.info('Labeling...')
        # Assurez-vous que les labels sont générés sur le dataframe de base (1min) avant le merge
        # ou que le merge est fait de manière à conserver l'index 1min
        # Pour l'instant, on génère les labels sur df_feat (qui a l'index 1min)
        # et on s'assure que les colonnes OHLCV 1min sont présentes pour le labeling
        required_label_cols = ['1min_<CLOSE>', '1min_<HIGH>', '1min_<LOW>']
        if not all(col in df_feat.columns for col in required_label_cols):
             logger.error("Colonnes OHLCV 1min manquantes pour le labeling après feature engineering multi-timeframe.")
             sys.exit(1)

        labels = generate_labels(df_feat, horizon=args.label_horizon, k=args.label_k)
        df_feat['label'] = labels
        df_feat = df_feat.dropna(subset=['label'])
        df_feat.to_pickle('artf_labeled.pkl')

        # 4. Balancing
        logger.info('Balancing...')
        df_bal = balance_by_block(df_feat, label_col='label', block_duration=args.block_duration, min_block_size=args.min_block_size)
        df_bal.to_pickle('artf_balanced.pkl')

        # 5. Splitting
        logger.info('Splitting...')
        train, val, test = split_dataset(
            df_bal,
            method=args.split_method,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            purge_window=args.purge_window
        )
        train.to_pickle('artf_train.pkl')
        val.to_pickle('artf_val.pkl')
        test.to_pickle('artf_test.pkl')

        # 6. Modeling
        logger.info('Modeling...')
        # Supprimer la colonne 'label' pour le modeling
        X_train = train.drop(columns=['label'])
        y_train = train['label']
        X_val = val.drop(columns=['label'])
        y_val = val['label']
        X_test = test.drop(columns=['label'])
        y_test = test['label']

        model = train_model(X_train, y_train, X_val, y_val, method=args.tuning, early_stopping_rounds=args.early_stopping_rounds)

        # 7. Evaluation
        logger.info('Evaluation...')
        y_pred = predict(model, X_test)

        class_metrics = evaluate_classification(y_test.values, y_pred)
        print('--- Classification report ---')
        print(class_metrics['classification_report'])

        # PnL/Sharpe
        # Utiliser la colonne de retour log du timeframe 1min pour le calcul du PnL
        returns_col = '1min_log_return' # Le nom de la colonne sera préfixé par le timeframe
        if returns_col in X_test.columns:
            returns = X_test[returns_col].values
        else:
            logger.warning(f"Colonne de retour log 1min ({returns_col}) non trouvée dans X_test. Calcul du PnL basé sur des retours nuls.")
            returns = np.zeros_like(y_pred)

        fin_metrics = evaluate_financial(y_pred, returns, index=X_test.index)
        print('--- Financial metrics ---')
        print({k: v for k, v in fin_metrics.items() if k != 'pnl_cum_curve'})

        if args.plot:
            logger.info('Tracé de la courbe de PnL...')
            fig = plot_pnl_curve(fin_metrics['pnl_cum_curve'])
            fig.savefig('artf_pnl_curve.png')

        # 8. Export
        if args.export:
            logger.info('Export des objets finaux...')
            # Assurez-vous de sauvegarder également le scaler si utilisé dans le modeling
            # Pour l'instant, le scaler n'est pas explicitement géré ici, juste le modèle
            save_objects(model, None, None, None, args.model_prefix)

        # 9. Inference (optionnel)
        if args.inference:
            logger.info('Inference sur le test set...')
            # L'inférence doit aussi utiliser les features multi-timeframe
            # Actuellement, inference prend X_test directement, ce qui est correct
            sig = inference(X_test, path_prefix=args.model_prefix)
            sig.to_csv('artf_inference_signal.csv')

        logger.info('Pipeline terminé avec succès.')

    except Exception as e:
        logger.error(f"Erreur critique dans le pipeline : {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 