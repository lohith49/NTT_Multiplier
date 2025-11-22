#!/usr/bin/env python3
"""
Main training script for FFT algorithm selection on fft_performance_results.csv
- Uses only features: Polynomial_Size, Sparsity, Dist_To_Next_Pow2, Is_Power_2, Is_Power_4
- Target: Best_Algorithm
- Models: XGBoost and RandomForest
- Fast training with solid fixed hyperparameters (no tuning)
- Assumes input dataset is balanced (no class weighting or SMOTE)
- Saves models and encoder alongside results
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

try:
    import xgboost as xgb
except Exception as e:
    print('xgboost import failed:', e)
    xgb = None

import joblib

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(__file__)

# Prefer GA-balanced dataset if present
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, 'fft_performance_results.csv')
BALANCED_DATASET_PATH = os.path.join(BASE_DIR, 'fft_clean_extradata.csv')
DATASET_PATH = BALANCED_DATASET_PATH if os.path.exists(BALANCED_DATASET_PATH) else DEFAULT_DATASET_PATH

SAVE_DIR = BASE_DIR

FEATURE_COLUMNS = ['Polynomial_Size', 'Sparsity', 'Dist_To_Next_Pow2', 'Is_Power_2', 'Is_Power_4']
TARGET_COLUMN = 'Best_Algorithm'

SCALE_FEATURES = False   # set True to apply StandardScaler


class FFTImbalanceTrainer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data: pd.DataFrame = None
        self.label_encoder = LabelEncoder()
        self.xgb_model = None
        self.rf_model = None
        self.best_model: Tuple[str, object] = None
        self.scaler = None

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

        # Ensure boolean-like columns are numeric (0/1)
        for col in ['Is_Power_2', 'Is_Power_4']:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            else:
                # Try to coerce strings like 'True'/'False' or 0/1 floats
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].map({'True':1, 'False':0}))
                df[col] = df[col].astype(int)

        # Ensure numeric columns
        df['Polynomial_Size'] = pd.to_numeric(df['Polynomial_Size'], errors='coerce')
        df['Sparsity'] = pd.to_numeric(df['Sparsity'], errors='coerce')
        df['Dist_To_Next_Pow2'] = pd.to_numeric(df['Dist_To_Next_Pow2'], errors='coerce')

        if df[[c for c in FEATURE_COLUMNS]].isnull().any().any():
            print('Warning: NaNs detected in features; dropping rows with NaNs')
            df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN], inplace=True)

        self.data = df
        return df

    def split_and_encode(self):
        X = self.data[FEATURE_COLUMNS].copy()
        y_text = self.data[TARGET_COLUMN].astype(str)
        y = self.label_encoder.fit_transform(y_text)

        # Report class distribution
        unique, counts = np.unique(y, return_counts=True)
        print('Class distribution (overall):')
        for cls, cnt in zip(unique, counts):
            label = self.label_encoder.inverse_transform([cls])[0]
            print(f"  {label}: {cnt} ({cnt / len(y) * 100:.2f}%)")

        # Stratified splits: train 70%, val 15%, test 15%
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
        )

        # Optionally scale numeric features (fit only on train)
        if SCALE_FEATURES:
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            X_train = pd.DataFrame(self.scaler.transform(X_train), columns=FEATURE_COLUMNS)
            X_val = pd.DataFrame(self.scaler.transform(X_val), columns=FEATURE_COLUMNS)
            X_test = pd.DataFrame(self.scaler.transform(X_test), columns=FEATURE_COLUMNS)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        if xgb is None:
            print('xgboost not installed; skipping XGBoost training')
            return

        # Strong fixed params for speed and accuracy (no search)
        params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'max_depth': 8,
            'eta': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'alpha': 0.1,
            'lambda': 1.2,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'seed': RANDOM_STATE,
        }
        num_boost_round = 800

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        print('Training XGBoost with early stopping...')
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=100,
        )

    def train_random_forest(self, X_train, y_train, X_val, y_val):
        print('Training RandomForest with fixed strong parameters...')
        rf = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,           # allow full growth
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        self.rf_model = rf

        # Simple sanity check on validation (accuracy only)
        val_pred = rf.predict(X_val)
        print(f"RF Val Accuracy: {accuracy_score(y_val, val_pred):.4f}")

    def evaluate(self, X_test, y_test):
        # XGBoost predictions
        if self.xgb_model is not None:
            dx = xgb.DMatrix(X_test)
            xgb_pred = np.argmax(self.xgb_model.predict(dx), axis=1)
        else:
            xgb_pred = None

        # RandomForest
        rf_pred = self.rf_model.predict(X_test) if self.rf_model is not None else None

        def report(name, y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            print(f"\n{name} Test Accuracy: {acc:.4f}")
            print('Classification report:')
            print(classification_report(
                y_true, y_pred, target_names=self.label_encoder.classes_, zero_division=0
            ))
            print('Confusion matrix:')
            print(confusion_matrix(y_true, y_pred))
            return acc

        results = {}
        if xgb_pred is not None:
            results['xgboost'] = report('XGBoost', y_test, xgb_pred)
        if rf_pred is not None:
            results['random_forest'] = report('RandomForest', y_test, rf_pred)

        # Choose best by accuracy
        winner = None
        best_metric = -1.0
        for name, acc in results.items():
            if acc > best_metric:
                best_metric = acc
                winner = name

        if winner:
            self.best_model = (winner, self.xgb_model if winner == 'xgboost' else self.rf_model)
            print(f"\nBest model: {winner} (Accuracy={best_metric:.4f})")

    def save(self):
        # Save XGBoost
        if self.xgb_model is not None:
            xgb_path = os.path.join(SAVE_DIR, 'xgboost_fft_model.json')
            try:
                self.xgb_model.save_model(xgb_path)
                print(f"Saved: {xgb_path}")
            except Exception as e:
                print('Failed to save XGBoost model:', e)

        # Save RandomForest
        if self.rf_model is not None:
            rf_path = os.path.join(SAVE_DIR, 'random_forest_fft_model.pkl')
            joblib.dump(self.rf_model, rf_path)
            print(f"Saved: {rf_path}")

        # Save encoder
        enc_path = os.path.join(SAVE_DIR, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, enc_path)
        print(f"Saved: {enc_path}")

        # Save scaler if used
        if self.scaler is not None:
            scaler_path = os.path.join(SAVE_DIR, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved: {scaler_path}")

        # Save best model info
        best_info = {
            'best_model_type': self.best_model[0] if self.best_model else None,
            'feature_names': FEATURE_COLUMNS,
            'target_name': TARGET_COLUMN,
        }
        info_path = os.path.join(SAVE_DIR, 'best_model_info.pkl')
        joblib.dump(best_info, info_path)
        print(f"Saved: {info_path}")


def main():
    print('FFT ML Training (balanced dataset, no tuning)')
    print('Dataset:', DATASET_PATH)

    trainer = FFTImbalanceTrainer(DATASET_PATH)
    trainer.load()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_and_encode()

    trainer.train_xgboost(X_train, y_train, X_val, y_val)
    trainer.train_random_forest(X_train, y_train, X_val, y_val)
    trainer.evaluate(X_test, y_test)
    trainer.save()

    print('\nDone.')


if __name__ == '__main__':
    main()
