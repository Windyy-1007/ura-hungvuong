"""
Simple Model Evaluation - No Unicode
===================================

Simple evaluation script without Unicode characters for Windows compatibility.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

def simple_evaluate():
    """Simple evaluation without Unicode issues"""
    print("Simple Model Evaluation")
    print("=" * 40)
    
    try:
        # Find model
        models_dir = Path("models")
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "multi_output" in d.name]
        if not model_dirs:
            print("No model found")
            return
        
        model_path = max(model_dirs, key=os.path.getmtime)
        model_file = model_path / "multi_output_medical_model.joblib"
        
        print(f"Model: {model_path.name}")
        print(f"Loading: {model_file}")
        
        # Load model
        model = joblib.load(model_file)
        
        # Load config
        with open("config_all_labels.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Load data
        sys.path.append('src/dataloader')
        from loader import load_data
        df = load_data("data/dataset_long_format_normalized_labeled.csv")
        
        print(f"Data: {len(df)} records, {len(df.columns)} columns")
        
        # Get features and targets
        exclude_features = config['feature_config']['exclude_features']
        target_prefixes = ['Kế hoạch (xử trí)']
        
        target_columns = [col for col in df.columns 
                         if any(col.startswith(prefix) for prefix in target_prefixes)]
        feature_columns = [col for col in df.columns 
                          if col not in target_columns and col not in exclude_features]
        
        print(f"Features: {len(feature_columns)}")
        print(f"Targets: {len(target_columns)}")
        
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_columns].copy()
        
        # Handle missing values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Filter valid targets
        valid_targets = []
        for col in target_columns:
            if col in y.columns:
                value_counts = y[col].value_counts()
                if len(value_counts) >= 2:
                    min_count = value_counts.min()
                    max_count = value_counts.max()
                    if min_count >= 10 and max_count / min_count <= 50:
                        valid_targets.append(col)
        
        y = y[valid_targets]
        print(f"Valid targets: {len(valid_targets)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.iloc[:, 0]
        )
        
        print(f"Test set: {len(X_test)} samples")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results = []
        for i, target in enumerate(valid_targets):
            target_name = target.replace('Kế hoạch (xử trí)_', '')
            
            y_true_target = y_test.iloc[:, i]
            y_pred_target = y_pred[:, i] if y_pred.ndim > 1 else y_pred
            
            accuracy = accuracy_score(y_true_target, y_pred_target)
            precision = precision_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
            recall = recall_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
            f1 = f1_score(y_true_target, y_pred_target, average='weighted', zero_division=0)
            
            results.append({
                'Target': target_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })
        
        # Print results
        print(f"\nRESULTS")
        print("=" * 60)
        print(f"{'Target':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        total_acc = 0
        total_prec = 0
        total_rec = 0
        total_f1 = 0
        
        for result in results:
            print(f"{result['Target']:<25} {result['Accuracy']:<10.3f} {result['Precision']:<10.3f} {result['Recall']:<10.3f} {result['F1_Score']:<10.3f}")
            total_acc += result['Accuracy']
            total_prec += result['Precision']
            total_rec += result['Recall']
            total_f1 += result['F1_Score']
        
        print("-" * 60)
        n_results = len(results)
        print(f"{'AVERAGE':<25} {total_acc/n_results:<10.3f} {total_prec/n_results:<10.3f} {total_rec/n_results:<10.3f} {total_f1/n_results:<10.3f}")
        
        print(f"\nSUMMARY")
        print(f"Average Accuracy: {total_acc/n_results:.1%}")
        print(f"Total Labels: {n_results}")
        print(f"Test Samples: {len(X_test)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_evaluate()
