#!/usr/bin/env python3
"""
FFT Algorithm Predictor using XGBoost and Random Forest
Predicts the best FFT algorithm (radix-2, radix-4, radix-split) based on input features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class FFTAlgorithmPredictor:
    def __init__(self, dataset_path):
        """Initialize the predictor with dataset path"""
        self.dataset_path = dataset_path
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.xgb_model = None
        self.rf_model = None
        self.best_model = None
        self.feature_names = ['polynomial_size', 'sparsity', 'power_of_2', 'power_of_4']
        
    def load_and_explore_data(self):
        """Load data and perform exploratory data analysis"""
        print("Loading dataset...")
        self.data = pd.read_csv(self.dataset_path)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Target: 'best' (algorithm choice)")
        
        # Display basic info
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(self.data.info())
        
        print("\n" + "="*50)
        print("FEATURE STATISTICS")
        print("="*50)
        print(self.data.describe())
        
        print("\n" + "="*50)
        print("TARGET DISTRIBUTION")
        print("="*50)
        target_counts = self.data['best'].value_counts()
        print(target_counts)
        print(f"\nTarget percentages:")
        print((target_counts / len(self.data) * 100).round(2))
        
        # Check for missing values
        print("\n" + "="*50)
        print("MISSING VALUES")
        print("="*50)
        missing_values = self.data.isnull().sum()
        print(missing_values)
        
        return self.data
    
    def visualize_data(self):
        """Create visualizations for data exploration"""
        plt.figure(figsize=(15, 12))
        
        # Target distribution
        plt.subplot(2, 3, 1)
        self.data['best'].value_counts().plot(kind='bar')
        plt.title('Target Distribution (Algorithm Choice)')
        plt.xticks(rotation=45)
        
        # Feature distributions
        for i, feature in enumerate(self.feature_names):
            plt.subplot(2, 3, i+2)
            self.data[feature].hist(bins=30, alpha=0.7)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        # Correlation heatmap
        plt.subplot(2, 3, 6)
        # Create a copy for correlation (encode categorical variable)
        data_corr = self.data.copy()
        data_corr['best_encoded'] = self.label_encoder.fit_transform(data_corr['best'])
        corr_matrix = data_corr[self.feature_names + ['best_encoded']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Algorithm choice by features
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i//2, i%2]
            for algorithm in self.data['best'].unique():
                subset = self.data[self.data['best'] == algorithm]
                ax.hist(subset[feature], alpha=0.7, label=algorithm, bins=20)
            ax.set_title(f'{feature} by Algorithm Choice')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('feature_by_algorithm.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_data(self):
        """Prepare data for training with proper encoding and splitting"""
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Separate features and target
        X = self.data[self.feature_names].copy()
        y = self.data['best'].copy()
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Label encoding mapping:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {label} -> {i}")
        
        # First split: 70% train, 30% temp (15% val + 15% test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, 
            test_size=0.3, 
            random_state=42, 
            stratify=y_encoded
        )
        
        # Second split: 15% val, 15% test from the 30% temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=0.5, 
            random_state=42, 
            stratify=y_temp
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Check class distribution in each set
        print("\nClass distribution in training set:")
        train_dist = pd.Series(y_train).value_counts().sort_index()
        for i, count in train_dist.items():
            print(f"  {self.label_encoder.classes_[i]}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(self):
        """Train XGBoost model with early stopping and overfitting prevention"""
        print("\n" + "="*50)
        print("TRAINING XGBOOST MODEL")
        print("="*50)
        
        # XGBoost parameters with overfitting prevention
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,          # Random sampling of training data
            'colsample_bytree': 0.8,   # Random sampling of features
            'reg_alpha': 0.1,          # L1 regularization
            'reg_lambda': 1.0,         # L2 regularization
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        
        # Training with early stopping
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        print("Training XGBoost with early stopping...")
        self.xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Make predictions on validation set
        dval_pred = xgb.DMatrix(self.X_val)
        val_pred_proba = self.xgb_model.predict(dval_pred)
        val_pred = np.argmax(val_pred_proba, axis=1)
        
        # Calculate validation accuracy
        val_accuracy = accuracy_score(self.y_val, val_pred)
        print(f"\nXGBoost Validation Accuracy: {val_accuracy:.4f}")
        
        return self.xgb_model
    
    def train_random_forest(self):
        """Train Random Forest model with overfitting prevention"""
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*50)
        
        # Random Forest with overfitting prevention measures
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,           # Limit tree depth
            'min_samples_split': 10,   # Minimum samples to split
            'min_samples_leaf': 5,     # Minimum samples in leaf
            'max_features': 'sqrt',    # Feature sampling
            'bootstrap': True,         # Bootstrap sampling
            'random_state': 42,
            'n_jobs': -1
        }
        
        print("Training Random Forest...")
        self.rf_model = RandomForestClassifier(**rf_params)
        
        # Fit the model
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Validation predictions
        val_pred = self.rf_model.predict(self.X_val)
        val_accuracy = accuracy_score(self.y_val, val_pred)
        print(f"Random Forest Validation Accuracy: {val_accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Random Forest):")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.rf_model
    
    def evaluate_models(self):
        """Evaluate both models and compare performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # XGBoost predictions
        dtest = xgb.DMatrix(self.X_test)
        xgb_pred_proba = self.xgb_model.predict(dtest)
        xgb_pred = np.argmax(xgb_pred_proba, axis=1)
        
        # Random Forest predictions
        rf_pred = self.rf_model.predict(self.X_test)
        rf_pred_proba = self.rf_model.predict_proba(self.X_test)
        
        # Calculate metrics
        xgb_accuracy = accuracy_score(self.y_test, xgb_pred)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        
        print(f"XGBoost Test Accuracy: {xgb_accuracy:.4f}")
        print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")
        
        # Detailed classification reports
        print("\n" + "-"*30)
        print("XGBOOST CLASSIFICATION REPORT")
        print("-"*30)
        print(classification_report(self.y_test, xgb_pred, 
                                  target_names=self.label_encoder.classes_))
        
        print("\n" + "-"*30)
        print("RANDOM FOREST CLASSIFICATION REPORT")
        print("-"*30)
        print(classification_report(self.y_test, rf_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Confusion matrices
        self.plot_confusion_matrices(xgb_pred, rf_pred)
        
        # Select best model
        if xgb_accuracy > rf_accuracy:
            self.best_model = ('xgboost', self.xgb_model)
            print(f"\n🏆 Best Model: XGBoost (Accuracy: {xgb_accuracy:.4f})")
        else:
            self.best_model = ('random_forest', self.rf_model)
            print(f"\n🏆 Best Model: Random Forest (Accuracy: {rf_accuracy:.4f})")
        
        return {
            'xgb_accuracy': xgb_accuracy,
            'rf_accuracy': rf_accuracy,
            'best_model': self.best_model[0]
        }
    
    def plot_confusion_matrices(self, xgb_pred, rf_pred):
        """Plot confusion matrices for both models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # XGBoost confusion matrix
        cm_xgb = confusion_matrix(self.y_test, xgb_pred)
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_, ax=axes[0])
        axes[0].set_title('XGBoost Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Random Forest confusion matrix
        cm_rf = confusion_matrix(self.y_test, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_, ax=axes[1])
        axes[1].set_title('Random Forest Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_best_algorithm(self, polynomial_size, sparsity, power_of_2, power_of_4):
        """Predict the best algorithm for given input parameters"""
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # Prepare input data
        input_data = np.array([[polynomial_size, sparsity, power_of_2, power_of_4]])
        
        model_type, model = self.best_model
        
        if model_type == 'xgboost':
            dinput = xgb.DMatrix(input_data)
            prediction_proba = model.predict(dinput)
            prediction = np.argmax(prediction_proba, axis=1)[0]
            confidence = np.max(prediction_proba)
        else:  # random_forest
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)
            confidence = np.max(prediction_proba)
        
        predicted_algorithm = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_algorithm, confidence
    
    def save_models(self):
        """Save trained models and encoders"""
        print("\n" + "="*50)
        print("SAVING MODELS")
        print("="*50)
        
        # Save XGBoost model
        self.xgb_model.save_model('xgboost_fft_model.json')
        print("✓ XGBoost model saved as 'xgboost_fft_model.json'")
        
        # Save Random Forest model
        joblib.dump(self.rf_model, 'random_forest_fft_model.pkl')
        print("✓ Random Forest model saved as 'random_forest_fft_model.pkl'")
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        print("✓ Label encoder saved as 'label_encoder.pkl'")
        
        # Save best model info
        best_model_info = {
            'best_model_type': self.best_model[0],
            'feature_names': self.feature_names
        }
        joblib.dump(best_model_info, 'best_model_info.pkl')
        print("✓ Best model info saved as 'best_model_info.pkl'")
    
    def cross_validate_models(self):
        """Perform cross-validation to check for overfitting"""
        print("\n" + "="*50)
        print("CROSS-VALIDATION ANALYSIS")
        print("="*50)
        
        # Combine train and validation sets for cross-validation
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = np.concatenate([self.y_train, self.y_val])
        
        # Cross-validation for Random Forest
        rf_cv_scores = cross_val_score(self.rf_model, X_combined, y_combined, cv=5, scoring='accuracy')
        print(f"Random Forest CV Scores: {rf_cv_scores}")
        print(f"Random Forest CV Mean ± Std: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
        
        # For XGBoost, we'll use a simpler approach due to its different API
        print(f"\nXGBoost validation performance was monitored during training with early stopping.")
        
        return rf_cv_scores


def main():
    """Main function to run the complete pipeline"""
    print("🚀 FFT Algorithm Predictor")
    print("="*60)
    
    # Initialize predictor
    predictor = FFTAlgorithmPredictor('fft_Dataset.csv')
    
    # Load and explore data
    predictor.load_and_explore_data()
    predictor.visualize_data()
    
    # Prepare data
    predictor.prepare_data()
    
    # Train models
    predictor.train_xgboost()
    predictor.train_random_forest()
    
    # Cross-validation
    predictor.cross_validate_models()
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Save models
    predictor.save_models()
    
    # Example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    test_cases = [
        (128, 0.5, 1, 0),
        (256, 0.25, 1, 1),
        (64, 0.75, 0, 0),
        (512, 0.1, 1, 0)
    ]
    
    for polynomial_size, sparsity, power_of_2, power_of_4 in test_cases:
        predicted_algo, confidence = predictor.predict_best_algorithm(
            polynomial_size, sparsity, power_of_2, power_of_4
        )
        print(f"Input: size={polynomial_size}, sparsity={sparsity}, pow2={power_of_2}, pow4={power_of_4}")
        print(f"  → Predicted: {predicted_algo} (confidence: {confidence:.3f})")
    
    print(f"\n✅ Training completed! Best model: {results['best_model']}")
    print(f"📊 Best accuracy: {max(results['xgb_accuracy'], results['rf_accuracy']):.4f}")

if __name__ == "__main__":
    main()