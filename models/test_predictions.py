"""
Test Script for Medical Prediction Models
=========================================

This script demonstrates how to use the exported medical prediction models
to make predictions for patient assessments and treatment plans.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the models directory to path to import the medical predictor
models_dir = Path(__file__).parent / "medical_models_20250908_210133"
sys.path.insert(0, str(models_dir))

from medical_predictor import MedicalPredictor

def test_medical_predictor():
    """Test the medical predictor with sample patient data."""
    
    print("🏥 Medical Prediction System Test")
    print("=" * 50)
    
    # Initialize the predictor
    try:
        predictor = MedicalPredictor(str(models_dir))
        print(f"\n✅ Medical predictor loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load predictor: {e}")
        return
    
    # Sample patient data (using realistic medical values)
    sample_patients = [
        {
            # Patient 1: Normal case
            "Mạch (nhập số nguyên)": 80,
            "HA tâm thu (nhập số nguyên)": 120,
            "HA tâm trương (nhập số nguyên)": 80,
            "Nhiệt độ (nhập số nguyên)": 37,
            "TT cơ bản (nhập số nguyên)": 140,
            "Các cơn co TC/10 phút (điền số nguyên)": 3,
            "Thời gian của các cơn co TC (điền số nguyên)": 40,
            "Cổ TC (KH: X)": 6,
            "Độ lọt (KH: O)": -2,
            "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 5
        },
        {
            # Patient 2: High-risk case
            "Mạch (nhập số nguyên)": 110,
            "HA tâm thu (nhập số nguyên)": 160,
            "HA tâm trương (nhập số nguyên)": 100,
            "Nhiệt độ (nhập số nguyên)": 38,
            "TT cơ bản (nhập số nguyên)": 100,
            "Các cơn co TC/10 phút (điền số nguyên)": 5,
            "Thời gian của các cơn co TC (điền số nguyên)": 60,
            "Cổ TC (KH: X)": 2,
            "Độ lọt (KH: O)": -3,
            "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 8
        }
    ]
    
    print(f"\n🔬 Testing predictions for {len(sample_patients)} sample patients...")
    
    for i, patient_data in enumerate(sample_patients, 1):
        print(f"\n{'='*60}")
        print(f"PATIENT {i} PREDICTIONS")
        print(f"{'='*60}")
        
        # Get all predictions
        predictions = predictor.predict(patient_data, return_probabilities=True)
        
        # Separate assessment and treatment predictions
        assessment_preds = {k: v for k, v in predictions.items() if 'Nhận định' in k and not k.endswith('_probability')}
        treatment_preds = {k: v for k, v in predictions.items() if 'Kế hoạch' in k and not k.endswith('_probability')}
        
        print(f"\n📋 ASSESSMENT PREDICTIONS:")
        print("-" * 40)
        for label, prediction in assessment_preds.items():
            prob_key = f"{label}_probability"
            probability = predictions.get(prob_key, 0.0)
            short_label = label.split('_', 1)[1] if '_' in label else label
            status = "✅ POSITIVE" if prediction == 1 else "❌ NEGATIVE"
            print(f"  {short_label:<30} {status} ({probability:.3f})")
        
        print(f"\n🏥 TREATMENT PREDICTIONS:")
        print("-" * 40)
        for label, prediction in treatment_preds.items():
            prob_key = f"{label}_probability"
            probability = predictions.get(prob_key, 0.0)
            short_label = label.split('_', 1)[1] if '_' in label else label
            status = "✅ RECOMMENDED" if prediction == 1 else "❌ NOT NEEDED"
            print(f"  {short_label:<30} {status} ({probability:.3f})")
        
        # Highlight high-priority predictions
        high_priority_assessments = [k for k, v in assessment_preds.items() if v == 1]
        high_priority_treatments = [k for k, v in treatment_preds.items() if v == 1]
        
        if high_priority_assessments or high_priority_treatments:
            print(f"\n⚠️  HIGH PRIORITY ITEMS:")
            if high_priority_assessments:
                print(f"   📋 Critical assessments: {len(high_priority_assessments)}")
                for item in high_priority_assessments[:3]:  # Show top 3
                    short_name = item.split('_', 1)[1] if '_' in item else item
                    print(f"      • {short_name}")
            
            if high_priority_treatments:
                print(f"   🏥 Required treatments: {len(high_priority_treatments)}")
                for item in high_priority_treatments[:3]:  # Show top 3
                    short_name = item.split('_', 1)[1] if '_' in item else item
                    print(f"      • {short_name}")
    
    # Test batch prediction
    print(f"\n{'='*60}")
    print("BATCH PREDICTION TEST")
    print(f"{'='*60}")
    
    df_patients = pd.DataFrame(sample_patients)
    batch_results = predictor.predict_batch(df_patients, return_probabilities=False)
    
    print(f"\n📊 Batch prediction completed for {len(df_patients)} patients")
    print(f"Result columns: {len(batch_results.columns)} predictions per patient")
    
    # Show summary statistics
    assessment_cols = [col for col in batch_results.columns if 'Nhận định' in col and not col.endswith('_probability')]
    treatment_cols = [col for col in batch_results.columns if 'Kế hoạch' in col and not col.endswith('_probability')]
    
    print(f"\n📈 BATCH RESULTS SUMMARY:")
    print(f"   Assessment predictions per patient: {batch_results[assessment_cols].sum(axis=1).tolist()}")
    print(f"   Treatment recommendations per patient: {batch_results[treatment_cols].sum(axis=1).tolist()}")
    
    # Model information
    print(f"\n{'='*60}")
    print("MODEL INFORMATION")
    print(f"{'='*60}")
    
    model_info = predictor.get_model_info()
    export_info = model_info['export_info']
    
    print(f"\n📊 Loaded Models Information:")
    print(f"   Total models: {export_info['total_models']}")
    print(f"   Export date: {export_info['export_date']}")
    print(f"   Assessment models: {len([m for m in model_info['models'].keys() if 'Nhận định' in m])}")
    print(f"   Treatment models: {len([m for m in model_info['models'].keys() if 'Kế hoạch' in m])}")
    
    print(f"\n🎉 Medical prediction system test completed successfully!")
    print(f"📋 All {export_info['total_models']} models are working correctly")

if __name__ == "__main__":
    test_medical_predictor()
