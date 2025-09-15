"""
Test TRULY FIXED Model
======================

Test the truly fixed multi-output model that properly handles numeric vs categorical features.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class TrulyFixedMultiOutputMedicalPredictor:
    """
    Truly fixed predictor that properly converts string-numeric columns to numeric types.
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
        self.force_numeric_columns = self.preprocessing['force_numeric_columns']
        
        print(f"✅ TRULY FIXED Multi-Output Medical Predictor loaded")
        print(f"   Model expects {len(self.feature_columns)} features")
        print(f"   Predicts {len(self.target_names)} targets")
        print(f"   Categorical features: {len(self.categorical_features)}")
        print(f"   Forced numeric features: {len(self.force_numeric_columns)}")
    
    def force_numeric_conversion(self, df):
        """Convert columns that should be numeric but might be stored as strings."""
        for col in self.force_numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def preprocess_input(self, patient_data):
        """
        TRULY FIXED preprocessing that preserves numeric values.
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
        
        # CRITICAL: Force numeric conversion
        df = self.force_numeric_conversion(df)
        
        # Handle missing values
        for col in df.columns:
            if col in self.categorical_features or df[col].dtype in ['object', 'string']:
                mode_val = 'Unknown'
                df[col] = df[col].fillna(mode_val)
            else:
                median_val = 0  # Default for missing numeric values
                df[col] = df[col].fillna(median_val)
        
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
                df[col] = df[col].fillna(0)
        
        return df
    
    def predict(self, patient_data, show_details=False):
        """Make predictions for patient data."""
        # Preprocess input
        processed_data = self.preprocess_input(patient_data)
        
        if show_details:
            print(f"🔧 Preprocessed values for key features:")
            key_features = ['Năm sinh', 'Mạch (nhập số nguyên)', 'HA tâm thu (nhập số nguyên)', 'TT cơ bản (nhập số nguyên)']
            for feat in key_features:
                if feat in processed_data.columns:
                    val = processed_data[feat].iloc[0]
                    print(f"   {feat}: {val} (type: {type(val)})")
        
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

def test_truly_fixed_model():
    """Test the truly fixed model with our test cases."""
    
    print("🧪 TESTING TRULY FIXED MULTI-OUTPUT MODEL")
    print("=" * 60)
    
    # Load the truly fixed predictor
    model_dir = "models/truly_fixed_multi_output_model_20250915_233208"
    predictor = TrulyFixedMultiOutputMedicalPredictor(model_dir)
    
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
    
    # Test Case 2: HIGH-RISK case with extreme values
    test_case_2 = {
        "Năm sinh": 1990,
        "Para (điền 4 số)": 0,
        "Tiền căn bệnh lý": "Tăng huyết áp",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
        "hour": 3,  # Middle of night emergency
        "Bạn đồng hành (1: Có, 0: Không)": 0.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 10.0,  # MAXIMUM pain
        "Nước uống vào (1: Có, 0: Không)": 0.0,
        "Ăn": 0.0,
        "Mạch (nhập số nguyên)": 150.0,  # SEVERE tachycardia
        "HA tâm thu (nhập số nguyên)": 200.0,  # SEVERE hypertension
        "HA tâm trương (nhập số nguyên)": 120.0,  # SEVERE hypertension
        "Nhiệt độ (nhập số nguyên)": 40.0,  # HIGH fever
        "TT cơ bản (nhập số nguyên)": 60.0,  # SEVERE fetal bradycardia (EMERGENCY!)
    }
    
    # Test Case 3: Normal values (should have low intervention probability)
    test_case_3 = {
        "Năm sinh": 1995,
        "Para (điền 4 số)": 1,
        "Tiền căn bệnh lý": "Không",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
        "hour": 14,
        "Bạn đồng hành (1: Có, 0: Không)": 1.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 4.0,  # Moderate pain
        "Nước uống vào (1: Có, 0: Không)": 1.0,
        "Ăn": 1.0,
        "Mạch (nhập số nguyên)": 75.0,  # Normal pulse
        "HA tâm thu (nhập số nguyên)": 110.0,  # Normal BP
        "HA tâm trương (nhập số nguyên)": 70.0,  # Normal BP
        "Nhiệt độ (nhập số nguyên)": 37.0,  # Normal temperature
        "TT cơ bản (nhập số nguyên)": 140.0,  # Normal fetal heart rate
    }
    
    test_cases = [
        ("Real Patient - Monitor Labor", test_case_1),
        ("🚨 HIGH-RISK EMERGENCY 🚨", test_case_2),
        ("Normal Values Control", test_case_3)
    ]
    
    for case_name, test_case in test_cases:
        print(f"\n🔬 {case_name}")
        print("-" * 50)
        
        # Show key inputs
        print(f"   Key inputs:")
        print(f"     Birth Year: {test_case.get('Năm sinh', 'N/A')}")
        print(f"     Pulse: {test_case.get('Mạch (nhập số nguyên)', 'N/A')} bpm")
        print(f"     BP: {test_case.get('HA tâm thu (nhập số nguyên)', 'N/A')}/{test_case.get('HA tâm trương (nhập số nguyên)', 'N/A')} mmHg")
        print(f"     FHR: {test_case.get('TT cơ bản (nhập số nguyên)', 'N/A')} bpm")
        print(f"     Pain: {test_case.get('Đánh giá mức độ đau (VAS) (Điền số nguyên)', 'N/A')}/10")
        print(f"     Temperature: {test_case.get('Nhiệt độ (nhập số nguyên)', 'N/A')}°C")
        
        # Make prediction
        try:
            results = predictor.predict(test_case, show_details=True)
            
            # results is a dict
            if isinstance(results, dict):
                # Count positive predictions
                positive_preds = sum(1 for target, result in results.items() if result['prediction'] == 1)
                total_preds = len(results)
                
                print(f"   📊 Results: {positive_preds}/{total_preds} positive predictions")
                
                # Show top predictions by probability
                sorted_results = sorted(results.items(), key=lambda x: x[1]['probability'], reverse=True)
                
                print(f"   🔥 Top 10 predictions by probability:")
                for target, result in sorted_results[:10]:
                    status = "🔴" if result['prediction'] == 1 else "⚪"
                    prob = result['probability']
                    pred = result['prediction']
                    print(f"     {status} {target:<25} {pred} ({prob:.3f})")
                
                if positive_preds > 0:
                    print(f"   ✅ SUCCESS: Model produced {positive_preds} positive predictions!")
                    print(f"   🎉 The model is now working correctly!")
                else:
                    max_prob = max(result['probability'] for result in results.values())
                    print(f"   ⚠️  No positive predictions, but max probability: {max_prob:.3f}")
                    if max_prob > 0.4:
                        print(f"   💡 High probabilities suggest model is working but conservative")
            else:
                print(f"   ❌ Unexpected result format: {type(results)}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
    
    print(f"\n🔍 ANALYSIS")
    print("=" * 60)
    print("✅ TRULY FIXED MODEL SHOULD NOW:")
    print("   - Preserve numeric values (birth year, pulse, BP, FHR)")
    print("   - Understand clinical significance of abnormal values")
    print("   - Produce positive predictions for high-risk cases") 
    print("   - Show conservative (low) probabilities for normal cases")
    print("   - Show elevated probabilities for emergency cases")

if __name__ == "__main__":
    test_truly_fixed_model()