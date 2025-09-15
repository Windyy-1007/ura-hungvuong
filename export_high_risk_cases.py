"""
Export High-Risk Test Cases
===========================

This script exports the high-risk test cases that successfully trigger 
positive predictions from the truly fixed model to a CSV file.
"""

import pandas as pd
import csv
from pathlib import Path

def export_high_risk_test_cases():
    """Export the high-risk test cases to CSV format."""
    
    print("🚨 EXPORTING HIGH-RISK TEST CASES")
    print("=" * 50)
    
    # Define the high-risk test cases that trigger positive predictions
    high_risk_cases = [
        {
            "Case_ID": "HIGH_RISK_1",
            "Clinical_Description": "Severe preeclampsia, fetal distress, prolonged labor",
            "Năm sinh": 1990,  # 35 years old
            "Para (điền 4 số)": 0,  # Nullipara (first pregnancy)
            "Tiền căn bệnh lý": "Tăng huyết áp",  # Hypertension history
            "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
            "hour": 3,  # Middle of night emergency
            "Bạn đồng hành (1: Có, 0: Không)": 0.0,  # No support person
            "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 9.0,  # Severe pain
            "Nước uống vào (1: Có, 0: Không)": 0.0,
            "Ăn": 0.0,
            "Mạch (nhập số nguyên)": 120.0,  # Tachycardia
            "HA tâm thu (nhập số nguyên)": 180.0,  # SEVERE hypertension
            "HA tâm trương (nhập số nguyên)": 110.0,  # SEVERE hypertension
            "Nhiệt độ (nhập số nguyên)": 38.5,  # Fever
            "Nước tiểu": "P+",  # Protein in urine (preeclampsia)
            "TT cơ bản (nhập số nguyên)": 80.0,  # Fetal bradycardia - EMERGENCY
            "CTG": "III",  # Pathological CTG
            "Nước ối (V: ối vỡ/Vg: Vàng)": "Vg",  # Meconium stained liquor
            "Kiểu thế": "CC PN",  # Occiput posterior
            "Bướu HT": "+",
            "Chồng khớp": "+",
            "Các cơn co TC/10 phút (điền số nguyên)": 6.0,  # Hyperstimulation
            "Thời gian của các cơn co TC (điền số nguyên)": 90.0,  # Prolonged contractions
            "Cổ TC (KH: X)": "8.0",
            "Độ lọt (KH: O)": "-1.0",
            "Oxytocin (số hoặc số la mã)": "V",  # High dose oxytocin
            "Thuốc": 1.0,
            "Truyền dịch": 1.0,
            "Expected_Report_Doctor": 1,
            "Expected_Any_Resuscitation": 1,
            "Expected_Fetal_Resuscitation": 1,
            "Risk_Level": "CRITICAL"
        },
        {
            "Case_ID": "HIGH_RISK_2", 
            "Clinical_Description": "Grand multipara, previous PPH history, hypotension",
            "Năm sinh": 1980,  # 45 years old
            "Para (điền 4 số)": 4004,  # Grand multipara
            "Tiền căn bệnh lý": "Xuất huyết sau sinh lần trước",  # Previous PPH
            "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
            "hour": 22,  # Late night
            "Bạn đồng hành (1: Có, 0: Không)": 1.0,
            "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 8.0,  # Severe pain
            "Nước uống vào (1: Có, 0: Không)": 0.0,
            "Ăn": 0.0,
            "Mạch (nhập số nguyên)": 110.0,  # Tachycardia
            "HA tâm thu (nhập số nguyên)": 95.0,  # HYPOTENSION - hemorrhage risk
            "HA tâm trương (nhập số nguyên)": 60.0,  # HYPOTENSION
            "Nhiệt độ (nhập số nguyên)": 36.0,
            "Nước tiểu": "A-",
            "TT cơ bản (nhập số nguyên)": 165.0,  # Fetal tachycardia
            "CTG": "II",  # Suspicious CTG
            "Nước ối (V: ối vỡ/Vg: Vàng)": "V",  # Ruptured membranes
            "Kiểu thế": "CC TN",
            "Bướu HT": "o",
            "Chồng khớp": "o", 
            "Các cơn co TC/10 phút (điền số nguyên)": 4.0,
            "Thời gian của các cơn co TC (điền số nguyên)": 50.0,
            "Cổ TC (KH: X)": "9.0",
            "Độ lọt (KH: O)": "0.0",
            "Oxytocin (số hoặc số la mã)": "",
            "Thuốc": 0.0,
            "Truyền dịch": 1.0,
            "Expected_Report_Doctor": 1,
            "Expected_Any_Resuscitation": 1,
            "Expected_Prevent_Hemorrhage": 1,
            "Risk_Level": "HIGH"
        },
        {
            "Case_ID": "HIGH_RISK_3",
            "Clinical_Description": "Fetal distress, hyperstimulation, emergency delivery needed", 
            "Năm sinh": 1995,  # 30 years old
            "Para (điền 4 số)": 1001,  # Second pregnancy
            "Tiền căn bệnh lý": "Không",
            "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
            "hour": 4,  # Emergency hours
            "Bạn đồng hành (1: Có, 0: Không)": 0.0,
            "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 10.0,  # MAXIMUM pain
            "Nước uống vào (1: Có, 0: Không)": 0.0,
            "Ăn": 0.0,
            "Mạch (nhập số nguyên)": 130.0,  # Severe tachycardia
            "HA tâm thu (nhập số nguyên)": 160.0,  # Hypertension
            "HA tâm trương (nhập số nguyên)": 100.0,  # Hypertension
            "Nhiệt độ (nhập số nguyên)": 39.0,  # High fever
            "Nước tiểu": "P-",
            "TT cơ bản (nhập số nguyên)": 60.0,  # SEVERE fetal bradycardia - EMERGENCY!
            "CTG": "III",  # Pathological CTG
            "Nước ối (V: ối vỡ/Vg: Vàng)": "Vg",  # Meconium
            "Kiểu thế": "CC PN",  # Malposition
            "Bướu HT": "+",
            "Chồng khớp": "+",
            "Các cơn co TC/10 phút (điền số nguyên)": 7.0,  # Hyperstimulation
            "Thời gian của các cơn co TC (điền số nguyên)": 120.0,  # Very prolonged
            "Cổ TC (KH: X)": "10.0",  # Fully dilated
            "Độ lọt (KH: O)": "+1.0",  # Engaged
            "Oxytocin (số hoặc số la mã)": "X",  # Maximum dose
            "Thuốc": 1.0,
            "Truyền dịch": 1.0,
            "Expected_Report_Doctor": 1,
            "Expected_Any_Resuscitation": 1,
            "Expected_Fetal_Resuscitation": 1,
            "Expected_Prepare_Delivery": 1,
            "Risk_Level": "CRITICAL"
        },
        {
            "Case_ID": "EXTREME_EMERGENCY",
            "Clinical_Description": "Multiple risk factors - eclampsia, severe fetal distress, hemorrhage",
            "Năm sinh": 1985,  # 40 years old
            "Para (điền 4 số)": 0,  # Elderly primigravida
            "Tiền căn bệnh lý": "Cao huyết áp, Tiểu đường thai kỳ",  # Multiple comorbidities
            "Khởi phát chuyển dạ (1: Có, 0: Không)": 1.0,
            "hour": 2,  # Middle of night
            "Bạn đồng hành (1: Có, 0: Không)": 0.0,
            "Đánh giá mức độ đau (VAS) (Điền số nguyên)": 10.0,  # Maximum pain
            "Nước uống vào (1: Có, 0: Không)": 0.0,
            "Ăn": 0.0,
            "Mạch (nhập số nguyên)": 150.0,  # Severe tachycardia
            "HA tâm thu (nhập số nguyên)": 200.0,  # SEVERE hypertension (eclampsia range)
            "HA tâm trương (nhập số nguyên)": 120.0,  # SEVERE hypertension
            "Nhiệt độ (nhập số nguyên)": 40.0,  # Very high fever
            "Nước tiểu": "P+++",  # Heavy proteinuria
            "TT cơ bản (nhập số nguyên)": 50.0,  # EXTREME fetal bradycardia
            "CTG": "III",  # Pathological
            "Nước ối (V: ối vỡ/Vg: Vàng)": "Vg",  # Thick meconium
            "Kiểu thế": "CC PN",  # Posterior position
            "Bướu HT": "++",  # Significant caput
            "Chồng khớp": "++",  # Significant molding
            "Các cơn co TC/10 phút (điền số nguyên)": 8.0,  # Severe hyperstimulation
            "Thời gian của các cơn co TC (điền số nguyên)": 150.0,  # Tetanic contractions
            "Cổ TC (KH: X)": "10.0",
            "Độ lọt (KH: O)": "+2.0",
            "Oxytocin (số hoặc số la mã)": "X",
            "Thuốc": 1.0,
            "Truyền dịch": 1.0,
            "Expected_Report_Doctor": 1,
            "Expected_Any_Resuscitation": 1,
            "Expected_Fetal_Resuscitation": 1,
            "Expected_Neonatal_Resuscitation": 1,
            "Expected_Prepare_Delivery": 1,
            "Expected_Prevent_Hemorrhage": 1,
            "Risk_Level": "EXTREME"
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(high_risk_cases)
    
    # Export to CSV
    output_file = "high_risk_test_cases.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ Exported {len(high_risk_cases)} high-risk test cases to: {output_file}")
    print(f"\n📋 Test Cases Summary:")
    for i, case in enumerate(high_risk_cases, 1):
        print(f"   {i}. {case['Case_ID']}: {case['Risk_Level']} risk")
        print(f"      Age: {2025 - case['Năm sinh']} years, Parity: {case['Para (điền 4 số)']}")
        print(f"      BP: {case['HA tâm thu (nhập số nguyên)']}/{case['HA tâm trương (nhập số nguyên)']} mmHg")
        print(f"      FHR: {case['TT cơ bản (nhập số nguyên)']} bpm")
        print(f"      CTG: Group {case['CTG']}")
        print(f"      Clinical: {case['Clinical_Description']}")
        print()
    
    # Also create a simplified version for model testing
    test_data = []
    for case in high_risk_cases:
        # Remove metadata columns for model input
        model_input = {k: v for k, v in case.items() 
                      if not k.startswith('Expected_') and 
                         k not in ['Case_ID', 'Clinical_Description', 'Risk_Level']}
        test_data.append(model_input)
    
    test_df = pd.DataFrame(test_data)
    test_output_file = "high_risk_model_input.csv"
    test_df.to_csv(test_output_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ Also created model input file: {test_output_file}")
    print(f"\n🔥 KEY FEATURES OF HIGH-RISK CASES:")
    print(f"   - Severe hypertension (180-200/110-120 mmHg)")
    print(f"   - Fetal bradycardia (50-80 bpm) - EMERGENCY signs")
    print(f"   - Pathological CTG (Group III)")
    print(f"   - Maximum pain scores (9-10/10)")
    print(f"   - Meconium-stained liquor")
    print(f"   - Grand multipara or elderly primigravida")
    print(f"   - Multiple risk factors combined")
    
    return output_file, test_output_file

if __name__ == "__main__":
    export_high_risk_test_cases()