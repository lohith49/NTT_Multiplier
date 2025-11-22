#!/usr/bin/env python3
"""
Simple prediction script for FFT Algorithm Predictor
Use this script to predict the best algorithm for new input parameters
"""

import numpy as np
import joblib
import xgboost as xgb
import argparse
import os
import pandas as pd


def load_models():
    """Load saved models and encoders"""
    try:
        # Resolve paths relative to this script's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        best_model_info_path = os.path.join(base_dir, 'best_model_info.pkl')
        label_encoder_path = os.path.join(base_dir, 'label_encoder.pkl')
        xgb_model_path = os.path.join(base_dir, 'xgboost_fft_model.json')
        rf_model_path = os.path.join(base_dir, 'random_forest_fft_model.pkl')

        # Load best model info
        best_model_info = joblib.load(best_model_info_path)
        label_encoder = joblib.load(label_encoder_path)
        
        best_model_type = best_model_info['best_model_type']
        feature_names = best_model_info['feature_names']
        
        if best_model_type == 'xgboost':
            model = xgb.Booster()
            model.load_model(xgb_model_path)
        else:  # random_forest
            model = joblib.load(rf_model_path)
        
        return model, label_encoder, best_model_type, feature_names
    
    except ModuleNotFoundError as e:
        print("Error: Missing dependency while loading model artifacts:", e)
        print("Please install required packages, e.g.: 'pip install scikit-learn xgboost'")
        return None, None, None, None
    except FileNotFoundError as e:
        print("Error: Model files not found. Please train the models first by running 'python fft_algorithm_predictor.py'")
        print(f"Missing file: {e}")
        return None, None, None, None


def predict_algorithm(model, label_encoder, model_type, polynomial_size, sparsity, dist_to_next_pow2, is_power_2, is_power_4, feature_names=None):
    """Predict the best algorithm for given parameters"""
    
    # Prepare input data in the same feature order used in training
    input_row = [polynomial_size, sparsity, dist_to_next_pow2, is_power_2, is_power_4]
    input_data = np.array([input_row])

    if model_type == 'xgboost':
        # XGBoost Booster expects DMatrix - use feature names if available
        if feature_names:
            dinput = xgb.DMatrix(input_data, feature_names=feature_names)
        else:
            dinput = xgb.DMatrix(input_data)
        
        raw_pred = model.predict(dinput)
        # Handle binary and multiclass outputs robustly
        if raw_pred.ndim == 1:
            prob_pos = float(raw_pred[0])
            prediction = int(prob_pos >= 0.5)
            confidence = prob_pos if prediction == 1 else 1.0 - prob_pos
        else:
            class_proba = raw_pred[0]
            prediction = int(np.argmax(class_proba))
            confidence = float(class_proba[prediction])
    else:  # random_forest
        # Use numpy array directly to match training format
        prediction = model.predict(input_data)[0]
        proba_row = model.predict_proba(input_data)[0]
        # Map predicted label to its probability using classes_
        try:
            class_index = list(model.classes_).index(prediction)
            confidence = float(proba_row[class_index])
        except Exception:
            confidence = float(np.max(proba_row))
    
    predicted_algorithm = label_encoder.inverse_transform([prediction])[0]
    
    return predicted_algorithm, confidence


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Predict the best FFT algorithm')
    parser.add_argument('--polynomial_size', type=int, required=True, 
                       help='Polynomial size (e.g., 128, 256, 512)')
    parser.add_argument('--sparsity', type=float, required=True,
                       help='Sparsity value between 0 and 1 (e.g., 0.25, 0.5, 0.75)')
    parser.add_argument('--dist_to_next_pow2', type=int, required=True,
                       help='Distance to next power of 2')
    parser.add_argument('--is_power_2', type=int, choices=[0, 1], required=True,
                       help='Is power of 2 flag (0 or 1)')
    parser.add_argument('--is_power_4', type=int, choices=[0, 1], required=True,
                       help='Is power of 4 flag (0 or 1)')
    
    args = parser.parse_args()
    
    # Load models
    model, label_encoder, model_type, feature_names = load_models()
    if model is None:
        return
    
    # Make prediction
    predicted_algorithm, confidence = predict_algorithm(
        model, label_encoder, model_type,
        args.polynomial_size, args.sparsity, args.dist_to_next_pow2, 
        args.is_power_2, args.is_power_4,
        feature_names=feature_names
    )
    
    # Display results
    print("="*50)
    print("FFT ALGORITHM PREDICTION")
    print("="*50)
    print(f"Input Parameters:")
    print(f"  Polynomial Size: {args.polynomial_size}")
    print(f"  Sparsity: {args.sparsity}")
    print(f"  Distance to Next Power of 2: {args.dist_to_next_pow2}")
    print(f"  Is Power of 2: {args.is_power_2}")
    print(f"  Is Power of 4: {args.is_power_4}")
    print(f"\nPrediction:")
    print(f"  🎯 Best Algorithm: {predicted_algorithm}")
    print(f"  📊 Confidence: {confidence:.3f}")
    print(f"  🤖 Model Used: {model_type}")


if __name__ == "__main__":
    # If no command line arguments, run interactive mode
    import sys
    if len(sys.argv) == 1:
        print("🎯 FFT Algorithm Predictor - Interactive Mode")
        print("="*50)
        
        # Load models
        model, label_encoder, model_type, feature_names = load_models()
        if model is None:
            sys.exit(1)
        
        try:
            while True:
                print("\nEnter parameters (or 'quit' to exit):")
                
                polynomial_size = input("Polynomial size: ")
                if polynomial_size.lower() == 'quit':
                    break
                polynomial_size = int(polynomial_size)
                
                sparsity = float(input("Sparsity (0-1): "))
                dist_to_next_pow2 = int(input("Distance to next power of 2: "))
                is_power_2 = int(input("Is power of 2 (0/1): "))
                is_power_4 = int(input("Is power of 4 (0/1): "))
                
                predicted_algorithm, confidence = predict_algorithm(
                    model, label_encoder, model_type,
                    polynomial_size, sparsity, dist_to_next_pow2, is_power_2, is_power_4,
                    feature_names=feature_names
                )
                
                print(f"\n🎯 Predicted Algorithm: {predicted_algorithm}")
                print(f"📊 Confidence: {confidence:.3f}")
                print("-" * 30)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
        except ValueError as e:
            print(f"Error: Invalid input - {e}")
    else:
        main()