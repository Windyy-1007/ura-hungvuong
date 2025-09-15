"""
Test Exported High-Risk Cases
=============================

Test the exported high-risk cases with our truly fixed model to verify 
they trigger positive predictions.
"""

import pandas as pd
from test_truly_fixed_model import TrulyFixedMultiOutputMedicalPredictor

def test_exported_high_risk_cases():
    """Test the exported high-risk cases."""
    
    print("🧪 TESTING EXPORTED HIGH-RISK CASES")
    print("=" * 60)
    
    # Load the truly fixed predictor
    model_dir = "models/truly_fixed_multi_output_model_20250915_233208"
    predictor = TrulyFixedMultiOutputMedicalPredictor(model_dir)
    
    # Load the exported test cases
    df = pd.read_csv("high_risk_model_input.csv")
    
    print(f"📋 Testing {len(df)} exported high-risk cases...\n")
    
    total_positive_predictions = 0
    case_number = 1
    
    for idx, row in df.iterrows():
        # Convert row to dict
        patient_data = row.to_dict()
        
        print(f"🔬 TEST CASE {case_number}")
        case_number += 1
        print("-" * 50)
        
        # Show key vital signs
        print(f"   Age: {2025 - patient_data['Năm sinh']} years")
        print(f"   Parity: {patient_data['Para (điền 4 số)']}")
        print(f"   BP: {patient_data['HA tâm thu (nhập số nguyên)']}/{patient_data['HA tâm trương (nhập số nguyên)']} mmHg")
        print(f"   Pulse: {patient_data['Mạch (nhập số nguyên)']} bpm")
        print(f"   FHR: {patient_data['TT cơ bản (nhập số nguyên)']} bpm")
        print(f"   Pain: {patient_data['Đánh giá mức độ đau (VAS) (Điền số nguyên)']}/10")
        print(f"   CTG: Group {patient_data['CTG']}")
        
        try:
            # Make prediction
            results = predictor.predict(patient_data)
            
            # Count positive predictions
            if isinstance(results, dict):
                positive_preds = sum(1 for target, result in results.items() if result['prediction'] == 1)
                total_positive_predictions += positive_preds
                
                print(f"   📊 Results: {positive_preds} positive predictions")
                
                if positive_preds > 0:
                    print(f"   ✅ POSITIVE PREDICTIONS:")
                    for target, result in results.items():
                        if result['prediction'] == 1:
                            prob = result['probability']
                            print(f"     🔴 {target}: {prob:.3f} confidence")
                
                # Show top probabilities even if not positive
                sorted_results = sorted(results.items(), key=lambda x: x[1]['probability'], reverse=True)
                print(f"   🔥 Top probabilities:")
                for target, result in sorted_results[:5]:
                    status = "🔴" if result['prediction'] == 1 else "⚪"
                    prob = result['probability']
                    print(f"     {status} {target}: {prob:.3f}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            print()
    
    print(f"🔍 SUMMARY")
    print("=" * 60)
    print(f"Total test cases: {len(df)}")
    print(f"Total positive predictions: {total_positive_predictions}")
    print(f"Average positive predictions per case: {total_positive_predictions / len(df):.1f}")
    
    if total_positive_predictions > 0:
        print(f"✅ SUCCESS: Model is responding to high-risk cases!")
    else:
        print(f"⚠️  Model appears conservative - may need threshold adjustment")

if __name__ == "__main__":
    test_exported_high_risk_cases()