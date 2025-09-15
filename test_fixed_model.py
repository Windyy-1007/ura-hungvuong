"""
Test Fixed Model with Real Patient Cases
========================================

This script tests the fixed multi-output model using our previously created
test cases to see if it now produces positive predictions for high-risk cases.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

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

def test_fixed_model():
    """Test the fixed model with our previously created test cases."""
    
    print("🧪 TESTING FIXED MULTI-OUTPUT MODEL")
    print("=" * 60)
    
    # Load the fixed predictor
    model_dir = "models/fixed_multi_output_medical_model_20250915_232829"
    predictor = FixedMultiOutputMedicalPredictor(model_dir)
    
    print(f"\n📋 TEST CASES")
    print("-" * 40)
    
    # Test Case 1: Real patient with Monitor_Labor = 1
    test_case_1 = {
        "Năm sinh": 1995,
        "Para (điền 4 số)": 0,
        "Tiền căn bệnh lý": "Không",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
        "hour": 14,
        "Bạn đồng hành (1: Có, 0: Không)": 1.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 3.0,
        "Nước uống vào (1: Có, 0: Không)": 0.0,
        "Ăn": 0.0,
        "Mạch (nhập số nguyên)": 90.0,
        "HA tâm thu (nhập số nguyên)": 120.0,
        "HA tâm trương (nhập số nguyên)": 80.0,
        "Nhiệt độ (nhập số nguyên)": 36.0,
        "TT cơ bản (nhập số nguyên)": 140.0,
    }
    
    # Test Case 2: High-risk hypertensive case
    test_case_2 = {
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
    
    # Test Case 3: Real patient with Report_Doctor = 1
    test_case_3 = {
        "Năm sinh": 1993,
        "Para (điền 4 số)": 1,
        "Tiền căn bệnh lý": "Không",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
        "hour": 22,
        "Bạn đồng hành (1: Có, 0: Không)": 1.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 8.0,
        "Nước uống vào (1: Có, 0: Không)": 0.0,
        "Ăn": 0.0,
        "Mạch (nhập số nguyên)": 100.0,
        "HA tâm thu (nhập số nguyên)": 130.0,
        "HA tâm trương (nhập số nguyên)": 85.0,
        "Nhiệt độ (nhập số nguyên)": 37.0,
        "TT cơ bản (nhập số nguyên)": 120.0,
    }
    
    test_cases = [
        ("Real Patient - Monitor Labor", test_case_1),
        ("High-Risk Hypertensive", test_case_2),
        ("Real Patient - Report Doctor", test_case_3)
    ]
    
    for case_name, test_case in test_cases:
        print(f"\n🔬 {case_name}")
        print("-" * 50)
        
        # Show key inputs
        print(f"   Key inputs:")
        print(f"     Pulse: {test_case.get('Mạch (nhập số nguyên)', 'N/A')}")
        print(f"     BP: {test_case.get('HA tâm thu (nhập số nguyên)', 'N/A')}/{test_case.get('HA tâm trương (nhập số nguyên)', 'N/A')}")
        print(f"     FHR: {test_case.get('TT cơ bản (nhập số nguyên)', 'N/A')}")
        print(f"     Pain: {test_case.get('Đánh giá mức độ đau (VAS) (Điền số nguyên)', 'N/A')}")
        
        # Make prediction
        try:
            results = predictor.predict(test_case)
            
            # results is a dict, not a list
            if isinstance(results, dict):
                # Count positive predictions
                positive_preds = sum(1 for target, result in results.items() if result['prediction'] == 1)
                total_preds = len(results)
                
                print(f"   📊 Results: {positive_preds}/{total_preds} positive predictions")
                
                # Show top predictions by probability
                sorted_results = sorted(results.items(), key=lambda x: x[1]['probability'], reverse=True)
                
                print(f"   🔥 Top predictions:")
                for target, result in sorted_results[:10]:
                    status = "🔴" if result['prediction'] == 1 else "⚪"
                    prob = result['probability']
                    pred = result['prediction']
                    print(f"     {status} {target:<25} {pred} ({prob:.3f})")
                
                if positive_preds > 0:
                    print(f"   ✅ SUCCESS: Model produced {positive_preds} positive predictions!")
                else:
                    print(f"   ❌ ISSUE: No positive predictions (all zeros)")
            else:
                print(f"   ❌ Unexpected result format: {type(results)}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
    
    print(f"\n🔍 SUMMARY")
    print("=" * 60)
    print("The fixed model should now:")
    print("✅ Preserve numeric values (pulse, BP, FHR)")
    print("✅ Only encode truly categorical features") 
    print("✅ Produce meaningful positive predictions for high-risk cases")

if __name__ == "__main__":
    test_fixed_model()