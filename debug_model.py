"""
Debug Model Predictions
======================

This script debugs why the model is predicting all zeros.
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

def debug_model_predictions():
    """Debug the prediction process step by step"""
    
    print("🔍 DEBUGGING MODEL PREDICTIONS")
    print("=" * 50)
    
    # Load model and preprocessing
    model_path = Path("models/multi_output_medical_model_20250910_165814")
    
    print("📁 Loading model components...")
    model = joblib.load(model_path / "multi_output_medical_model.joblib")
    preprocessing = joblib.load(model_path / "preprocessing.joblib")
    
    print(f"✅ Model loaded: {type(model)}")
    print(f"✅ Preprocessing loaded: {len(preprocessing)} components")
    
    # Load original training data to understand distribution
    print("\n📊 Loading original training data...")
    df = pd.read_csv("data/dataset_long_format_normalized_labeled.csv")
    
    # Get target distributions
    target_columns = [col for col in df.columns if col.startswith('Kế hoạch (xử trí)_')]
    print(f"\n🎯 Target Label Distributions:")
    print("-" * 40)
    
    target_stats = {}
    for col in target_columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            if len(value_counts) >= 2:
                pos_ratio = value_counts.get(1, 0) / len(df) * 100
                target_stats[col] = {
                    'total_samples': len(df),
                    'positive_cases': value_counts.get(1, 0),
                    'positive_ratio': pos_ratio
                }
                print(f"{col.replace('Kế hoạch (xử trí)_', ''):<25} {value_counts.get(1, 0):>4}/{len(df):<4} ({pos_ratio:.1f}%)")
    
    # Create a test case that should definitely be positive
    print(f"\n🧪 Creating extreme test case...")
    extreme_case = {
        "Năm sinh": 1990,
        "Para (điền 4 số)": 0,
        "Tiền căn bệnh lý": "Tăng huyết áp",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
        "hour": 20,
        "Bạn đồng hành (1: Có, 0: Không)": 0.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 10.0,  # Maximum pain
        "Nước uống vào (1: Có, 0: Không)": 0.0,
        "Ăn": 0.0,
        "Mạch (nhập số nguyên)": 150.0,  # Severe tachycardia
        "HA tâm thu (nhập số nguyên)": 200.0,  # Severe hypertension
        "HA tâm trương (nhập số nguyên)": 120.0,
        "Nhiệt độ (nhập số nguyên)": 40.0,  # High fever
        "TT cơ bản (nhập số nguyên)": 50.0,  # Severe fetal bradycardia
    }
    
    # Debug preprocessing step by step
    print(f"\n🔧 DEBUGGING PREPROCESSING")
    print("-" * 40)
    
    # Step 1: Create DataFrame
    test_df = pd.DataFrame([extreme_case])
    print(f"1. Test data shape: {test_df.shape}")
    print(f"   Sample values: Pulse={extreme_case['Mạch (nhập số nguyên)']}, BP={extreme_case['HA tâm thu (nhập số nguyên)']}")
    
    # Step 2: Add missing feature columns
    feature_columns = preprocessing['feature_columns']
    print(f"2. Expected features: {len(feature_columns)}")
    
    for col in feature_columns:
        if col not in test_df.columns:
            test_df[col] = np.nan
    
    test_df = test_df[feature_columns]
    print(f"   After adding missing columns: {test_df.shape}")
    
    # Step 3: Handle missing values
    print(f"3. Handling missing values...")
    missing_before = test_df.isna().sum().sum()
    
    for col in test_df.columns:
        if test_df[col].dtype in ['object', 'string']:
            test_df[col].fillna('Unknown', inplace=True)
        else:
            test_df[col].fillna(test_df[col].median() if not test_df[col].isna().all() else 0, inplace=True)
    
    missing_after = test_df.isna().sum().sum()
    print(f"   Missing values: {missing_before} -> {missing_after}")
    
    # Step 4: Apply label encoders
    print(f"4. Applying label encoders...")
    label_encoders = preprocessing['label_encoders']
    print(f"   Available encoders: {len(label_encoders)}")
    
    # Check for problematic encoders
    numeric_cols_with_encoders = []
    for col, encoder in label_encoders.items():
        if col in test_df.columns:
            original_dtype = test_df[col].dtype
            print(f"   Encoding {col}: dtype={original_dtype}, classes={len(encoder.classes_)}")
            
            # Check if this is incorrectly encoding a numeric column
            if original_dtype in ['int64', 'float64']:
                numeric_cols_with_encoders.append(col)
                print(f"   ⚠️  WARNING: Encoding numeric column {col}!")
            
            test_df[col] = test_df[col].astype(str)
            known_labels = set(encoder.classes_)
            unknown_mask = ~test_df[col].isin(known_labels)
            if unknown_mask.any():
                print(f"   ⚠️  Unknown values in {col}, using default: {encoder.classes_[0]}")
                test_df.loc[unknown_mask, col] = encoder.classes_[0]
            test_df[col] = encoder.transform(test_df[col])
    
    if numeric_cols_with_encoders:
        print(f"\n❌ PROBLEM IDENTIFIED: {len(numeric_cols_with_encoders)} numeric columns being encoded!")
        print(f"   Problematic columns: {numeric_cols_with_encoders[:5]}")
    
    # Step 5: Make prediction
    print(f"\n5. Making prediction...")
    print(f"   Preprocessed data shape: {test_df.shape}")
    print(f"   Sample preprocessed values: {test_df.iloc[0, :5].values}")
    
    # Get raw model output
    raw_prediction = model.predict(test_df)
    prediction_proba = model.predict_proba(test_df)
    
    print(f"   Raw prediction shape: {raw_prediction.shape}")
    print(f"   Raw prediction values: {raw_prediction[0]}")
    print(f"   Any positive predictions: {np.any(raw_prediction[0] == 1)}")
    
    # Check prediction probabilities
    print(f"\n📊 PREDICTION PROBABILITIES:")
    print("-" * 40)
    
    target_names = preprocessing.get('target_names', [f"Target_{i}" for i in range(len(raw_prediction[0]))])
    
    for i, (target, pred, proba_dist) in enumerate(zip(target_names, raw_prediction[0], prediction_proba)):
        if hasattr(proba_dist, '__iter__') and len(proba_dist) > 1:
            proba_positive = proba_dist[0][1] if len(proba_dist[0]) > 1 else 0.0
        else:
            proba_positive = 0.0
        
        status = "🔴" if pred == 1 else "⚪"
        print(f"   {status} {target:<25} Pred={pred} P(1)={proba_positive:.3f}")
    
    # Check against some actual positive cases from training data
    print(f"\n🔍 COMPARING WITH ACTUAL POSITIVE CASES")
    print("-" * 50)
    
    # Find a case with positive labels in the original data
    for target_col in target_columns[:3]:  # Check first 3 targets
        if target_col in df.columns:
            positive_cases = df[df[target_col] == 1]
            if len(positive_cases) > 0:
                print(f"\n📋 Analyzing actual positive case for {target_col.replace('Kế hoạch (xử trí)_', '')}:")
                
                # Get the first positive case
                positive_case = positive_cases.iloc[0]
                
                # Show key features
                key_features = ['Mạch (nhập số nguyên)', 'HA tâm thu (nhập số nguyên)', 'TT cơ bản (nhập số nguyên)', 'Đánh giá mức độ đau (VAS) (Điền số nguyên)']
                print(f"   Key features from positive case:")
                for feature in key_features:
                    if feature in positive_case:
                        print(f"     {feature}: {positive_case[feature]}")
                
                break
    
    print(f"\n🔍 DIAGNOSIS:")
    print("=" * 50)
    if numeric_cols_with_encoders:
        print("❌ MAJOR ISSUE: Numeric columns are being incorrectly label-encoded")
        print("   This corrupts the feature values and makes predictions unreliable")
        print("   Solution: Fix preprocessing to only encode categorical columns")
    
    if not np.any(raw_prediction[0] == 1):
        print("❌ MODEL ISSUE: No positive predictions even for extreme case")
        print("   This suggests model threshold issues or training data problems")
    
    print("\n💡 RECOMMENDED FIXES:")
    print("1. Fix preprocessing to preserve numeric columns")
    print("2. Retrain model with correct preprocessing")
    print("3. Verify training data has sufficient positive examples")

if __name__ == "__main__":
    debug_model_predictions()