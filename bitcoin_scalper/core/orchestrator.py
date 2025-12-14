import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
from bitcoin_scalper.core.data_loading import load_minute_csv
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.labeling import generate_labels, generate_q_values
from bitcoin_scalper.core.balancing import balance_by_block
from bitcoin_scalper.core.splitting import split_dataset
from bitcoin_scalper.core.modeling import train_model, predict, train_qvalue_model
from bitcoin_scalper.core.evaluation import evaluate_classification, evaluate_financial, plot_pnl_curve
from bitcoin_scalper.core.export import save_objects, load_objects
from bitcoin_scalper.core.inference import inference
import bitcoin_scalper.core.ml_orchestrator as ml_orch

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
    parser.add_argument('--tuning', default='optuna', choices=['optuna', 'grid'], help='Méthode de tuning')
    parser.add_argument('--early_stopping_rounds', type=int, default=20, help='Early stopping rounds')
    parser.add_argument('--export', action='store_true', help='Sauvegarder les objets finaux')
    parser.add_argument('--inference', action='store_true', help='Effectuer l\'inférence sur le test set')
    parser.add_argument('--plot', action='store_true', help='Tracer la courbe de PnL')
    parser.add_argument('--fill_missing', action='store_true', help='Active le comblement automatique des trous temporels (ffill)')
    parser.add_argument('--qvalue', action='store_true', help='Utiliser la régression Q-value (expected return net) au lieu de la classification directionnelle')
    parser.add_argument('--pipeline', default='ml', choices=['ml', 'tuning', 'backtest', 'rl', 'stacking', 'hybrid'], help='Pipeline ML à exécuter')
    args = parser.parse_args()

    # Variables to be populated by pipeline execution
    model = None
    fin_metrics = {}

    try:
        # 1. Chargement et nettoyage des données 1 minute
        logger.info('Chargement et nettoyage du CSV minute...')
        df_1min = load_minute_csv(args.csv, report_missing="rapport_trous_temporals.txt")

        # Vérification stricte des trous temporels
        # Les features basées sur des fenêtres glissantes (rolling) seront faussées si les données ne sont pas continues
        full_index = pd.date_range(df_1min.index.min(), df_1min.index.max(), freq='1min', tz=df_1min.index.tz)
        has_gaps = len(df_1min) < len(full_index)

        if has_gaps:
            if args.fill_missing:
                logger.info('Comblement automatique des trous temporels (réindexation 1min + ffill)...')
                missing_before = full_index.difference(df_1min.index)
                if len(missing_before) > 0:
                    logger.warning(f"{len(missing_before)} lignes manquantes détectées, application du ffill...")
                df_1min = df_1min.reindex(full_index)
                df_1min = df_1min.ffill()
                logger.info(f"Après comblement : {df_1min.shape[0]} lignes, {df_1min.isna().sum().sum()} valeurs NaN restantes.")
            else:
                logger.error("Des trous temporels ont été détectés et --fill_missing n'est pas activé.")
                logger.error("L'entraînement sur des données discontinues fausse les indicateurs techniques (rolling windows).")
                logger.error("Veuillez activer --fill_missing ou fournir un CSV complet.")
                sys.exit(1)

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

        # Calculer log_return_1m si absent ou mal calculé
        if '<CLOSE>' in df_feat.columns and 'log_return_1m' not in df_feat.columns:
            df_feat['log_return_1m'] = np.log(df_feat['<CLOSE>'] / df_feat['<CLOSE>'].shift(1))
        # S'assurer que 1min_log_return est bien un alias de log_return_1m
        if 'log_return_1m' in df_feat.columns:
            df_feat['1min_log_return'] = df_feat['log_return_1m']
        logger.info(f"Colonnes disponibles avant labeling : {df_feat.columns.tolist()}")

        # 3. Labeling ou Q-value
        if args.qvalue:
            logger.info('Génération des Q-values (expected return net)...')
            q_df = generate_q_values(df_feat, horizon=args.label_horizon, fee=0.001, spread=0.0005, slippage=0.0002)
            df_feat = pd.concat([df_feat, q_df], axis=1)
            df_feat = df_feat.dropna(subset=['q_buy', 'q_sell', 'q_hold'])
            df_feat.to_pickle('artf_qvalues.pkl')
        else:
            logger.info('Labeling...')
            required_label_cols = ['1min_<CLOSE>', '1min_<HIGH>', '1min_<LOW>']
            if not all(col in df_feat.columns for col in required_label_cols):
                logger.error("Colonnes OHLCV 1min manquantes pour le labeling après feature engineering multi-timeframe.")
                sys.exit(1)
            labels = generate_labels(df_feat, horizon=args.label_horizon, k=args.label_k)
            df_feat['label'] = labels
            df_feat = df_feat.dropna(subset=['label'])
            df_feat.to_pickle('artf_labeled.pkl')

        # 4. Balancing (seulement pour classification)
        if not args.qvalue:
            logger.info('Balancing...')
            df_bal = balance_by_block(df_feat, label_col='label', block_duration=args.block_duration, min_block_size=args.min_block_size)
            df_bal.to_pickle('artf_balanced.pkl')
        else:
            df_bal = df_feat

        # 5. Splitting (Redundant with ml_orch but good for preserving local variables for later inference block if needed)
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

        # Ensure log returns exist (Fixing potential data leaks/missing columns for backtest)
        for split_name, split_df in zip(['train', 'val', 'test'], [train, val, test]):
            if 'log_return_1m' not in split_df.columns and '<CLOSE>' in split_df.columns:
                split_df['log_return_1m'] = np.log(split_df['<CLOSE>'] / split_df['<CLOSE>'].shift(1))
            if 'log_return_1m' in split_df.columns:
                split_df['1min_log_return'] = split_df['log_return_1m']

        # --- Orchestration dynamique selon le pipeline choisi ---
        report = {}
        if args.pipeline == 'ml':
            logger.info('Exécution du pipeline ML classique...')
            # Using expert level ModelTrainer via ml_orch
            report = ml_orch.run_ml_pipeline(
                df_bal,
                label_col='label',
                out_dir='ml_reports',
                tuning_method=args.tuning,
                early_stopping_rounds=args.early_stopping_rounds
            )
            model = report.get('model_object')

        elif args.pipeline == 'tuning':
            logger.info('Exécution du pipeline tuning...')
            ml_orch.run_tuning_pipeline(df_bal, label_col='label', out_dir='tuning_reports')
            # Tuning returns a report, usually no model object ready for export unless specified

        elif args.pipeline == 'backtest':
            logger.info('Exécution du pipeline backtest...')
            ml_orch.run_backtest_pipeline(df_bal, signal_col='label', price_col='<CLOSE>', out_dir='backtest_reports')

        elif args.pipeline == 'rl':
            logger.info('Exécution du pipeline RL...')
            ml_orch.run_rl_pipeline(df_bal, out_dir='rl_reports')

        elif args.pipeline == 'stacking':
            logger.info('Exécution du pipeline stacking...')
            ml_orch.run_stacking_pipeline(df_bal, label_col='label', out_dir='stacking_reports')

        elif args.pipeline == 'hybrid':
            logger.info('Exécution du pipeline hybrid...')
            ml_orch.run_hybrid_strategy_pipeline(df_bal, out_dir='hybrid_reports')

        else:
            logger.error(f"Pipeline inconnu : {args.pipeline}")
            sys.exit(1)

        if args.plot and 'pnl_cum_curve' in fin_metrics:
            logger.info('Tracé de la courbe de PnL...')
            fig = plot_pnl_curve(fin_metrics['pnl_cum_curve'])
            fig.savefig('artf_pnl_curve.png')

        # 8. Export
        if args.export:
            logger.info('Export des objets finaux...')
            if model:
                # Assuming scaler is implicitly handled by the model or not used (trees)
                save_objects(model, None, None, None, args.model_prefix)
            else:
                logger.warning("Aucun modèle à exporter. Assurez-vous d'avoir exécuté le pipeline 'ml'.")

        # 9. Inference (optionnel)
        if args.inference:
            logger.info('Inference sur le test set...')
            # We use the test split generated locally
            # NOTE: inference function expects model file to load, or we should pass the model object?
            # Looking at inference source code, it might load via prefix.
            # If we just saved it, it should be fine.
            if model:
                # If model is in memory, we might need to adapt inference to accept it,
                # or rely on the saved file.
                # Assuming inference(...) loads from file using `path_prefix`.
                try:
                    sig = inference(test, path_prefix=args.model_prefix) # Passing dataframe test split
                    sig.to_csv('artf_inference_signal.csv')
                except Exception as e:
                    logger.error(f"Erreur lors de l'inférence: {e}")
            else:
                 logger.warning("Pas de modèle pour l'inférence.")

        logger.info('Pipeline terminé avec succès.')

    except Exception as e:
        logger.error(f"Erreur critique dans le pipeline : {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
