"""
Test Case for URA Medical Prediction Model
==========================================

This script demonstrates how to test the model with real patient data
that has at least one treatment label valued "1" in the original dataset.
"""

import pandas as pd
import sys
import os
from pathlib import Path

def create_test_case():
    """Create test case from real patient data with positive labels"""
    
    # Test Case 1: Patient with Monitor_Labor = 1 and Report_Doctor = 1
    # Based on real case from dataset (Lê Thị Thanh Nữ - row 5)
    test_case_1 = {
        # Demographics and timing
        "Năm sinh": 1993,
        "Para (điền 4 số)": 0,
        "Tiền căn bệnh lý": "",  # Empty in original
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 0.0,
        "hour": 15,
        
        # Pain and companions
        "Bạn đồng hành (1: Có, 0: Không)": 1.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 3.0,
        "Nước uống vào (1: Có, 0: Không)": 1.0,
        "Ăn": 0.0,
        
        # Vital signs
        "Mạch (nhập số nguyên)": 90.0,
        "HA tâm thu (nhập số nguyên)": 120.0,
        "HA tâm trương (nhập số nguyên)": 70.0,
        "Nhiệt độ (nhập số nguyên)": 37.0,
        "Nước tiểu": "P-",
        "TT cơ bản (nhập số nguyên)": 140.0,
        
        # Clinical measurements
        "CTG": "I",
        "Nước ối (V: ối vỡ/Vg: Vàng)": "Đ",
        "Kiểu thế": "CC TN",
        "Bướu HT": "o",
        "Chồng khớp": "o",
        "Các cơn co TC/10 phút (điền số nguyên)": 3.0,
        "Thời gian của các cơn co TC (điền số nguyên)": 40.0,
        "Cổ TC (KH: X)": 5.0,
        "Độ lọt (KH: O)": -3.0,
        "Oxytocin (số hoặc số la mã)": "",  # Empty
        "Thuốc": 0.0,
        "Truyền dịch": 1.0,
        
        # Expected predictions (from original data)
        "Expected_Monitor_Labor": 1,
        "Expected_Report_Doctor": 0,
        "Expected_Position_Unfavorable": 1,  # Assessment label (not predicted)
        "Clinical_Note": "Kiểu thế không thuận lợi - Position unfavorable"
    }
    
    # Test Case 2: Patient with Report_Doctor = 1 
    # Based on real case (NGUYỄN THỊ LOAN - row 6)
    test_case_2 = {
        # Demographics and timing  
        "Năm sinh": 1983,
        "Para (điền 4 số)": 2002,
        "Tiền căn bệnh lý": "Bình thường",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 0.0,
        "hour": 15,
        
        # Pain and companions
        "Bạn đồng hành (1: Có, 0: Không)": 1.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 3.0,
        "Nước uống vào (1: Có, 0: Không)": 1.0,
        "Ăn": 0.0,
        
        # Vital signs
        "Mạch (nhập số nguyên)": 90.0,
        "HA tâm thu (nhập số nguyên)": 120.0,
        "HA tâm trương (nhập số nguyên)": 70.0,
        "Nhiệt độ (nhập số nguyên)": 36.8,
        "Nước tiểu": "P-, A-",
        "TT cơ bản (nhập số nguyên)": 140.0,
        
        # Clinical measurements
        "CTG": "I",
        "Nước ối (V: ối vỡ/Vg: Vàng)": "V",
        "Kiểu thế": "CC TN",
        "Bướu HT": "o",
        "Chồng khớp": "o",
        "Các cơn co TC/10 phút (điền số nguyên)": 2.0,
        "Thời gian của các cơn co TC (điền số nguyên)": 30.0,
        "Cổ TC (KH: X)": 6.0,
        "Độ lọt (KH: O)": -3.0,
        "Oxytocin (số hoặc số la mã)": "VIII",
        "Thuốc": 0.0,
        "Truyền dịch": 0.0,
        
        # Expected predictions
        "Expected_Monitor_Labor": 0,
        "Expected_Report_Doctor": 1,
        "Expected_Position_Unfavorable": 1,
        "Clinical_Note": "Kiểu thế không thuận lợi, cần trình bác sĩ"
    }
    
    # Test Case 3: High-risk case with multiple interventions
    # Based on real case (Nguyễn Hồng Diễm - row 17)
    test_case_3 = {
        # Demographics and timing
        "Năm sinh": 1984,
        "Para (điền 4 số)": 3003,  # Multipara
        "Tiền căn bệnh lý": "Mổ nội soi ruột thừa viêm 2022",
        "Khởi phát chuyển dạ (1: Có, 0: Không)": 0.0,
        "hour": 17,
        
        # Pain and companions
        "Bạn đồng hành (1: Có, 0: Không)": 1.0,
        "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 3.0,
        "Nước uống vào (1: Có, 0: Không)": 0.0,
        "Ăn": 0.0,
        
        # Vital signs
        "Mạch (nhập số nguyên)": 80.0,
        "HA tâm thu (nhập số nguyên)": 120.0,
        "HA tâm trương (nhập số nguyên)": 70.0,
        "Nhiệt độ (nhập số nguyên)": 37.3,
        "Nước tiểu": "P-",
        "TT cơ bản (nhập số nguyên)": 150.0,
        
        # Clinical measurements
        "CTG": "II",  # CTG Group II - more concerning
        "Nước ối (V: ối vỡ/Vg: Vàng)": "T, Đ",
        "Kiểu thế": "CC PN",
        "Bướu HT": "o",
        "Chồng khớp": "o", 
        "Các cơn co TC/10 phút (điền số nguyên)": 5.0,
        "Thời gian của các cơn co TC (điền số nguyên)": 60.0,
        "Cổ TC (KH: X)": 5.0,
        "Độ lọt (KH: O)": -3.0,
        "Oxytocin (số hoặc số la mã)": "",
        "Thuốc": 0.0,
        "Truyền dịch": 0.0,
        
        # Expected predictions
        "Expected_Monitor_Labor": 1,
        "Expected_Prevent_Hemorrhage": 1,
        "Expected_Patient_Stable": 1,  # Assessment
        "Clinical_Note": "Đa sản, Thiếu máu - need hemorrhage prevention"
    }
    
    return [test_case_1, test_case_2, test_case_3]

def test_model_predictions():
    """Test the model with real patient cases"""
    
    print("URA Medical Model Test Cases")
    print("=" * 50)
    
    # Get test cases
    test_cases = create_test_case()
    
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
        
        print(f"📋 Testing {len(test_cases)} patient cases...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"TEST CASE {i}: {test_case['Clinical_Note']}")
            print("-" * 60)
            
            # Extract features (remove expected values and notes)
            patient_data = {k: v for k, v in test_case.items() 
                          if not k.startswith('Expected_') and k != 'Clinical_Note'}
            
            # Make prediction
            try:
                results = predictor.predict(patient_data)
                
                print("PATIENT DATA:")
                print(f"  Age: {2025 - test_case['Năm sinh']} years")
                print(f"  Parity: {test_case['Para (điền 4 số)']}")
                print(f"  Pulse: {test_case['Mạch (nhập số nguyên)']} bpm")
                print(f"  BP: {test_case['HA tâm thu (nhập số nguyên)']}/{test_case['HA tâm trương (nhập số nguyên)']} mmHg")
                print(f"  CTG: Group {test_case['CTG']}")
                print(f"  Pain Score: {test_case['Đánh giá mức độ đau (VAS) (Điền số nguyên)']}/10")
                
                print("\nPREDICTIONS:")
                
                # Show treatment predictions with probabilities  
                treatment_predictions = []
                if isinstance(results, dict):
                    for target_name, result in results.items():
                        if 'Monitor_Labor' in target_name or 'Report_Doctor' in target_name or 'Prepare_Delivery' in target_name or 'Resuscitation' in target_name:
                            prediction = result['prediction']
                            probability = result['probability']
                            
                            status = "🟢 YES" if prediction == 1 else "⚪ NO"
                            treatment_predictions.append((target_name, prediction, probability, status))
                
                # Sort by probability (highest first)
                treatment_predictions.sort(key=lambda x: x[2], reverse=True)
                
                # Show top predictions
                print("  Top Treatment Recommendations:")
                for target_name, prediction, probability, status in treatment_predictions[:5]:
                    print(f"    {status} {target_name:<25} ({probability:.1%} confidence)")
                
                # Check if any expected labels were predicted correctly
                expected_labels = [k for k in test_case.keys() if k.startswith('Expected_')]
                if expected_labels:
                    print("\nVERIFICATION:")
                    for expected_key in expected_labels:
                        if 'Monitor_Labor' in expected_key or 'Report_Doctor' in expected_key or 'Prevent_Hemorrhage' in expected_key:
                            treatment_name = expected_key.replace('Expected_', '')
                            
                            expected_value = test_case[expected_key]
                            # Find matching prediction
                            predicted_value = 0
                            if isinstance(results, dict):
                                for target_name, result in results.items():
                                    if treatment_name in target_name:
                                        predicted_value = result['prediction']
                                        break
                            
                            match_status = "✅ MATCH" if expected_value == predicted_value else "❌ DIFFER"
                            print(f"    {treatment_name}: Expected={expected_value}, Predicted={predicted_value} {match_status}")
                
                print("\n" + "="*60 + "\n")
                
            except Exception as e:
                print(f"❌ Error predicting case {i}: {e}")
                continue
        
        print("✅ Test cases completed!")
        
    except Exception as e:
        print(f"❌ Error setting up test: {e}")
        import traceback
        traceback.print_exc()

def save_test_cases_csv():
    """Save test cases as CSV for batch testing"""
    test_cases = create_test_case()
    
    # Remove expected values and notes for clean CSV
    clean_cases = []
    for case in test_cases:
        clean_case = {k: v for k, v in case.items() 
                     if not k.startswith('Expected_') and k != 'Clinical_Note'}
        clean_cases.append(clean_case)
    
    df = pd.DataFrame(clean_cases)
    
    # Save to CSV
    output_file = "test_cases_patient_data.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Test cases saved to: {output_file}")
    print(f"📊 Use with: python main.py --predict {output_file}")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test URA Medical Model')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save test cases as CSV file')
    
    args = parser.parse_args()
    
    if args.save_csv:
        save_test_cases_csv()
    else:
        test_model_predictions()