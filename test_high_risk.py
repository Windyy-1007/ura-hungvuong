"""
High-Risk Test Cases for URA Medical Model
==========================================

This script creates test cases designed to trigger positive predictions
by using more extreme clinical values that would require intervention.
"""

import pandas as pd
import sys
import os
from pathlib import Path

def create_high_risk_test_cases():
    """Create test cases with extreme values likely to trigger positive predictions"""
    
    # High-Risk Case 1: Severe hypertension + high pain + CTG abnormalities
    high_risk_case_1 = {
        "Năm sinh": 1990,
        "Para (điền 4 số)": 0,  # Primigravida
        "Tiền căn bệnh lý": "Tăng huyết áp",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
        "hour": 20,
        
        # High pain and stress indicators
        "Bạn đồng hành (1: Có, 0: Không)": 0.0,  # No companion
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 9.0,  # Severe pain
        "Nước uống vào (1: Có, 0: Không)": 0.0,
        "Ăn": 0.0,
        
        # Concerning vital signs
        "Mạch (nhập số nguyên)": 120.0,  # Tachycardia
        "HA tâm thu (nhập số nguyên)": 180.0,  # Severe hypertension
        "HA tâm trương (nhập số nguyên)": 110.0,  # Severe diastolic hypertension
        "Nhiệt độ (nhập số nguyên)": 38.5,  # Fever
        "Nước tiểu": "P+",  # Proteinuria
        "TT cơ bản (nhập số nguyên)": 80.0,  # Bradycardia fetal
        
        # Concerning clinical findings
        "CTG": "III",  # CTG Group III - most concerning
        "Nước ối (V: ối vỡ/Vg: Vàng)": "Vg",  # Meconium stained
        "Kiểu thế": "CC PN",  # Posterior position
        "Bướu HT": "++",  # Caput succedaneum
        "Chồng khớp": "++",  # Molding
        "Các cơn co TC/10 phút (điền số nguyên)": 6.0,  # Frequent contractions
        "Thời gian của các cơn co TC (điền số nguyên)": 90.0,  # Long contractions
        "Cổ TC (KH: X)": 3.0,  # Slow progress
        "Độ lọt (KH: O)": -4.0,  # High station
        "Oxytocin (số hoặc số la mã)": "X",  # High dose
        "Thuốc": 1.0,
        "Truyền dịch": 1.0,
        
        "Clinical_Note": "Severe preeclampsia, fetal distress, prolonged labor"
    }
    
    # High-Risk Case 2: Hemorrhage risk + multipara
    high_risk_case_2 = {
        "Năm sinh": 1980,
        "Para (điền 4 số)": 4004,  # Grand multipara
        "Tiền căn bệnh lý": "Băng huyết sau sinh lần trước",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
        "hour": 2,  # Night time
        
        # Risk factors
        "Bạn đồng hành (1: Có, 0: Không)": 1.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 8.0,
        "Nước uống vào (1: Có, 0: Không)": 1.0,
        "Ăn": 0.0,
        
        # Vital signs suggesting blood loss
        "Mạch (nhập số nguyên)": 110.0,  # Tachycardia
        "HA tâm thu (nhập số nguyên)": 95.0,  # Hypotension
        "HA tâm trương (nhập số nguyên)": 60.0,  # Low diastolic
        "Nhiệt độ (nhập số nguyên)": 36.0,  # Hypothermia
        "Nước tiểu": "P-, A-",
        "TT cơ bản (nhập số nguyên)": 165.0,  # Fetal tachycardia
        
        # Clinical findings
        "CTG": "II",  # CTG Group II
        "Nước ối (V: ối vỡ/Vg: Vàng)": "V",  # Ruptured membranes
        "Kiểu thế": "CC TN",
        "Bướu HT": "o",
        "Chồng khớp": "o",
        "Các cơn co TC/10 phút (điền số nguyên)": 2.0,  # Weak contractions
        "Thời gian của các cơn co TC (điền số nguyên)": 20.0,  # Short contractions
        "Cổ TC (KH: X)": 8.0,  # Nearly complete
        "Độ lọt (KH: O)": 1.0,  # Low station
        "Oxytocin (số hoặc số la mã)": "IV",
        "Thuốc": 1.0,
        "Truyền dịch": 1.0,
        
        "Clinical_Note": "Grand multipara, previous PPH history, hypotension"
    }
    
    # High-Risk Case 3: Emergency delivery case
    emergency_case = {
        "Năm sinh": 1995,
        "Para (điền 4 số)": 1001,
        "Tiền căn bệnh lý": "Bình thường",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
        "hour": 23,  # Late night
        
        # Emergency indicators
        "Bạn đồng hành (1: Có, 0: Không)": 1.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 10.0,  # Maximum pain
        "Nước uống vào (1: Có, 0: Không)": 0.0,
        "Ăn": 0.0,
        
        # Vital signs
        "Mạch (nhập số nguyên)": 130.0,  # Severe tachycardia
        "HA tâm thu (nhập số nguyên)": 160.0,  # Hypertension
        "HA tâm trương (nhập số nguyên)": 100.0,
        "Nhiệt độ (nhập số nguyên)": 39.0,  # High fever
        "Nước tiểu": "P+, A+",  # Proteinuria + ketones
        "TT cơ bản (nhập số nguyên)": 60.0,  # Severe fetal bradycardia
        
        # Emergency findings
        "CTG": "III",  # Pathological CTG
        "Nước ối (V: ối vỡ/Vg: Vàng)": "Vg",  # Thick meconium
        "Kiểu thế": "CC PN",  # Malposition
        "Bướu HT": "+++",  # Severe caput
        "Chồng khớp": "+++",  # Severe molding
        "Các cơn co TC/10 phút (điền số nguyên)": 7.0,  # Hyperstimulation
        "Thời gian của các cơn co TC (điền số nguyên)": 120.0,  # Tetanic contractions
        "Cổ TC (KH: X)": 10.0,  # Complete cervix
        "Độ lọt (KH: O)": 2.0,  # Low but not progressing
        "Oxytocin (số hoặc số la mã)": "XII",  # Very high dose
        "Thuốc": 1.0,
        "Truyền dịch": 1.0,
        
        "Clinical_Note": "Fetal distress, hyperstimulation, emergency delivery needed"
    }
    
    return [high_risk_case_1, high_risk_case_2, emergency_case]

def test_high_risk_cases():
    """Test the model with high-risk cases that should trigger positive predictions"""
    
    print("URA Medical Model - High-Risk Test Cases")
    print("=" * 60)
    print("Testing cases designed to trigger positive treatment predictions")
    print()
    
    # Get test cases
    test_cases = create_high_risk_test_cases()
    
    try:
        # Find the latest model
        models_dir = Path("models")
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "multi_output" in d.name]
        if not model_dirs:
            print("❌ No model found. Please train a model first.")
            return
        
        model_path = max(model_dirs, key=os.path.getmtime)
        print(f"🤖 Using model: {model_path.name}")
        
        # Load the truly fixed predictor
        from test_truly_fixed_model import TrulyFixedMultiOutputMedicalPredictor
        
        predictor = TrulyFixedMultiOutputMedicalPredictor(str(model_path))
        
        print(f"📋 Testing {len(test_cases)} high-risk cases...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"HIGH-RISK CASE {i}: {test_case['Clinical_Note']}")
            print("-" * 80)
            
            # Extract features (remove notes)
            patient_data = {k: v for k, v in test_case.items() if k != 'Clinical_Note'}
            
            # Make prediction
            try:
                results = predictor.predict(patient_data)
                
                print("CRITICAL PATIENT DATA:")
                print(f"  🚨 Age: {2025 - test_case['Năm sinh']} years")
                print(f"  🚨 Parity: {test_case['Para (điền 4 số)']}")
                print(f"  🚨 Blood Pressure: {test_case['HA tâm thu (nhập số nguyên)']}/{test_case['HA tâm trương (nhập số nguyên)']} mmHg")
                print(f"  🚨 Pulse: {test_case['Mạch (nhập số nguyên)']} bpm")
                print(f"  🚨 Temperature: {test_case['Nhiệt độ (nhập số nguyên)']}°C")
                print(f"  🚨 Fetal Heart Rate: {test_case['TT cơ bản (nhập số nguyên)']} bpm")
                print(f"  🚨 CTG: Group {test_case['CTG']}")
                print(f"  🚨 Pain Score: {test_case['Đánh giá mức độ đau (VAS) (Điền số nguyên)']}/10")
                
                print("\n🎯 TREATMENT PREDICTIONS:")
                
                # Get all treatment predictions
                treatment_predictions = []
                positive_predictions = []
                
                if isinstance(results, dict):
                    for target_name, result in results.items():
                        prediction = result['prediction']
                        probability = result['probability']
                        
                        if prediction == 1:
                            status = "🔴 REQUIRED"
                            positive_predictions.append((target_name, probability))
                        else:
                            status = "⚪ Not needed"
                        
                        treatment_predictions.append((target_name, prediction, probability, status))
                
                # Sort by probability (highest first)
                treatment_predictions.sort(key=lambda x: x[2], reverse=True)
                
                # Show all predictions with high probability
                high_prob_predictions = [p for p in treatment_predictions if p[2] > 0.3]
                
                if positive_predictions:
                    print(f"  ✅ POSITIVE PREDICTIONS ({len(positive_predictions)}):")
                    for treatment_name, probability in positive_predictions:
                        print(f"    🔴 {treatment_name:<25} ({probability:.1%} confidence)")
                else:
                    print("  ⚠️  NO POSITIVE PREDICTIONS")
                
                print(f"\n  📊 HIGH PROBABILITY RECOMMENDATIONS (>30%):")
                for treatment_name, prediction, probability, status in high_prob_predictions:
                    print(f"    {status} {treatment_name:<25} ({probability:.1%})")
                
                if not high_prob_predictions:
                    print("    ⚠️  No high-probability recommendations")
                
                print("\n" + "="*80 + "\n")
                
            except Exception as e:
                print(f"❌ Error predicting case {i}: {e}")
                continue
        
        print("✅ High-risk test cases completed!")
        
    except Exception as e:
        print(f"❌ Error setting up test: {e}")
        import traceback
        traceback.print_exc()

def save_high_risk_csv():
    """Save high-risk test cases as CSV"""
    test_cases = create_high_risk_test_cases()
    
    # Remove notes for clean CSV
    clean_cases = []
    for case in test_cases:
        clean_case = {k: v for k, v in case.items() if k != 'Clinical_Note'}
        clean_cases.append(clean_case)
    
    df = pd.DataFrame(clean_cases)
    
    # Save to CSV
    output_file = "high_risk_test_cases.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 High-risk test cases saved to: {output_file}")
    print(f"📊 Use with: python main.py --predict {output_file}")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test URA Medical Model with High-Risk Cases')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save high-risk test cases as CSV file')
    
    args = parser.parse_args()
    
    if args.save_csv:
        save_high_risk_csv()
    else:
        test_high_risk_cases()