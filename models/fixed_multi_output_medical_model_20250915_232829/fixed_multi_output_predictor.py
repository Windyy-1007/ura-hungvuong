"""
Fixed Multi-Output Medical Predictor
===================================

This file provides a simple interface to use the trained multi-output model.
Fixed version that properly handles numeric vs categorical features.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class FixedMultiOutputMedicalPredictor:
    """
    Fixed predictor that properly handles numeric vs categorical features.
    """
    
    def __init__(self, model_dir):
        """Load the trained model and preprocessing components."""
        self.model_dir = Path(model_dir)
        
        # Load model and preprocessing
        self.model = joblib.load(self.model_dir / "multi_output_medical_model.joblib")
        self.preprocessing = joblib.load(self.model_dir / "preprocessing.joblib")
        
        # Extract components
        self.label_encoders = self.preprocessing['label_encoders']
        self.feature_columns = self.preprocessing['feature_columns'] 
        self.target_names = self.preprocessing['target_names']
        self.categorical_features = self.preprocessing['categorical_features']
        
        print(f"✅ Fixed Multi-Output Medical Predictor loaded")
        print(f"   Model expects {len(self.feature_columns)} features")
        print(f"   Predicts {len(self.target_names)} targets")
        print(f"   Categorical features: {len(self.categorical_features)}")
    
    def preprocess_input(self, patient_data):
        """
        FIXED preprocessing that preserves numeric values.
        """
        # Convert to DataFrame if needed
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()
        
        # Add missing feature columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Select only required features
        df = df[self.feature_columns]
        
        # Handle missing values
        for col in df.columns:
            if col in self.categorical_features or df[col].dtype in ['object', 'string']:
                df[col].fillna('Unknown', inplace=True)
            else:
                df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
        
        # Apply label encoding ONLY to categorical features
        for col in self.categorical_features:
            if col in df.columns and col in self.label_encoders:
                encoder = self.label_encoders[col]
                df[col] = df[col].astype(str)
                
                # Handle unknown categories
                known_labels = set(encoder.classes_)
                unknown_mask = ~df[col].isin(known_labels)
                if unknown_mask.any():
                    df.loc[unknown_mask, col] = encoder.classes_[0]
                
                df[col] = encoder.transform(df[col])
        
        # Ensure numeric columns remain numeric
        for col in df.columns:
            if col not in self.categorical_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    df[col].fillna(0, inplace=True)
        
        return df
    
    def predict(self, patient_data):
        """Make predictions for patient data."""
        # Preprocess input
        processed_data = self.preprocess_input(patient_data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        # Format results
        results = []
        for i in range(len(processed_data)):
            result = {}
            for j, target_name in enumerate(self.target_names):
                pred = predictions[i][j]
                # Get probability of positive class
                if hasattr(probabilities[j][i], '__len__') and len(probabilities[j][i]) > 1:
                    prob = probabilities[j][i][1]
                else:
                    prob = 0.0
                
                result[target_name] = {
                    'prediction': int(pred),
                    'probability': float(prob)
                }
            results.append(result)
        
        return results[0] if len(results) == 1 else results

# Example usage
if __name__ == "__main__":
    # Load predictor
    predictor = FixedMultiOutputMedicalPredictor("fixed_multi_output_medical_model_20250915_232829")
    
    # Example prediction
    test_patient = {
        "Năm sinh": 1990,
        "Mạch (nhập số nguyên)": 120.0,
        "HA tâm thu (nhập số nguyên)": 140.0,
        "HA tâm trương (nhập số nguyên)": 90.0,
        "TT cơ bản (nhập số nguyên)": 120.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 7.0
    }
    
    results = predictor.predict(test_patient)
    
    print("\n🔮 Prediction Results:")
    for target, result in results.items():
        status = "🔴" if result['prediction'] == 1 else "⚪"
        print(f"   {status} {target:<25} {result['prediction']} ({result['probability']:.3f})")
