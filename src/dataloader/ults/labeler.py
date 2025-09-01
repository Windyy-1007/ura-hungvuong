import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_labels_from_text(text_series, column_name):
    """
    Extract binary labels from text values by identifying key medical terms.
    
    Parameters:
    text_series (pd.Series): Series containing text values
    column_name (str): Name of the column for specific label extraction
    
    Returns:
    pd.DataFrame: DataFrame with binary label columns
    """
    
    if column_name == "Nhận định và đánh giá":
        # Define label categories for Assessment and Evaluation
        label_definitions = {
            'CTG_Group_I': ['CTG nhóm I', 'ctg nhóm i', 'ctg i'],
            'CTG_Group_II': ['CTG nhóm II', 'ctg nhóm ii', 'ctg ii'],
            'CTG_Group_III': ['CTG nhóm III', 'ctg nhóm iii', 'ctg iii'],
            'Guidance_Push': ['Hướng dẫn rặn', 'huong dan ran', 'hướng dẫn rặn sanh'],
            'Patient_Stable': ['Sản phụ ổn', 'san phu on', 'sản phụ ổn', 'sp ổn'],
            'Position_Unfavorable': ['Kiểu thế không thuận lợi', 'kt không thuận lợi', 'kiểu thế không thuận lợi'],
            'Multipara': ['Đa sản', 'da san', 'đa sanh'],
            'Contractions_Complete': ['CTC trọn', 'ctc trọn', 'cơn co trọn'],
            'Contractions_Stopped': ['CTC ngưng tiến', 'ctc ngưng', 'cơn co ngưng'],
            'Fever': ['Sốt', 'sot', 'nhiệt độ cao'],
            'Pelvis_Disproportion': ['Gò chưa phù hợp', 'go chua phu hop'],
            'Not_Dangerous': ['chưa nguy hiểm', 'chua nguy hiem', 'không nguy hiểm'],
            'Monitor_Delivery': ['theo dõi sanh', 'td sanh', 'monitor delivery'],
            'Patient_Pain': ['đau nhiều', 'đau', 'pain'],
            'Fetal_Resuscitation': ['Hồi sức thai', 'hoi suc thai', 'resuscitation'],
            'Amniotic_Fluid_Poor': ['Ối xấu', 'oi xau', 'amniotic fluid poor']
        }
        
    elif column_name == "Kế hoạch (xử trí)":
        # Define label categories for Treatment Plan
        label_definitions = {
            'Monitor_Labor': ['Theo dõi tiếp chuyển dạ', 'theo doi tiep chuyen da', 'td tiếp cd'],
            'Report_Doctor': ['Trình bác sĩ', 'trinh bac si', 'trình bs', 'báo bác sĩ'],
            'Prepare_Delivery': ['Chuẩn bị dụng cụ sanh', 'chuan bi dung cu sanh', 'cb dc sanh'],
            'Neonatal_Resuscitation': ['HSSS', 'hồi sức sơ sinh', 'hsss'],
            'Prevent_Hemorrhage': ['Đề phòng băng huyết', 'de phong bang huyet', 'dpbh'],
            'Fetal_Resuscitation': ['Hồi sức thai', 'hoi suc thai', 'resuscitation'],
            'Notify_Attending': ['Báo bác sĩ thân chủ', 'bao bac si than chu'],
            'Reassess_Later': ['Đánh giá lại', 'danh gia lai', 'reassess'],
            'Continue_Resuscitation': ['Tiếp tục hồi sức', 'tiep tuc hoi suc'],
            'Monitor_Delivery': ['TD sanh', 'theo dõi sanh', 'monitor delivery']
        }
    else:
        raise ValueError(f"Unknown column: {column_name}")
    
    # Initialize DataFrame for labels
    labels_df = pd.DataFrame(index=text_series.index)
    
    # Create binary labels
    for label_name, patterns in label_definitions.items():
        label_col = f"{column_name}_{label_name}"
        labels_df[label_col] = 0
        
        # Check each text value against patterns
        for idx, text_value in text_series.items():
            if pd.notna(text_value):
                text_lower = str(text_value).lower()
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        labels_df.loc[idx, label_col] = 1
                        break
    
    return labels_df

def create_composite_labels(labels_df, column_prefix):
    """
    Create composite labels for combinations of medical conditions.
    
    Parameters:
    labels_df (pd.DataFrame): DataFrame with individual binary labels
    column_prefix (str): Prefix for the column names
    
    Returns:
    pd.DataFrame: Updated DataFrame with composite labels
    """
    
    if "Nhận định và đánh giá" in column_prefix:
        # Create composite labels for assessments
        
        # CTG with other conditions
        if f"{column_prefix}_CTG_Group_II" in labels_df.columns and f"{column_prefix}_Position_Unfavorable" in labels_df.columns:
            labels_df[f"{column_prefix}_CTG_II_Position_Unfavorable"] = (
                labels_df[f"{column_prefix}_CTG_Group_II"] & 
                labels_df[f"{column_prefix}_Position_Unfavorable"]
            ).astype(int)
        
        if f"{column_prefix}_CTG_Group_II" in labels_df.columns and f"{column_prefix}_Guidance_Push" in labels_df.columns:
            labels_df[f"{column_prefix}_CTG_II_Guidance"] = (
                labels_df[f"{column_prefix}_CTG_Group_II"] & 
                labels_df[f"{column_prefix}_Guidance_Push"]
            ).astype(int)
        
        # Any CTG abnormality (Group II or III)
        ctg_cols = [col for col in labels_df.columns if 'CTG_Group' in col and ('II' in col or 'III' in col)]
        if ctg_cols:
            labels_df[f"{column_prefix}_CTG_Abnormal"] = labels_df[ctg_cols].any(axis=1).astype(int)
    
    elif "Kế hoạch (xử trí)" in column_prefix:
        # Create composite labels for treatment plans
        
        # Delivery preparation with hemorrhage prevention
        if f"{column_prefix}_Prepare_Delivery" in labels_df.columns and f"{column_prefix}_Prevent_Hemorrhage" in labels_df.columns:
            labels_df[f"{column_prefix}_Delivery_Prep_Hemorrhage"] = (
                labels_df[f"{column_prefix}_Prepare_Delivery"] & 
                labels_df[f"{column_prefix}_Prevent_Hemorrhage"]
            ).astype(int)
        
        # Any resuscitation needed
        resus_cols = [col for col in labels_df.columns if 'Resuscitation' in col]
        if resus_cols:
            labels_df[f"{column_prefix}_Any_Resuscitation"] = labels_df[resus_cols].any(axis=1).astype(int)
        
        # Multiple interventions (3 or more different actions)
        intervention_cols = [col for col in labels_df.columns if column_prefix in col and 
                           any(term in col for term in ['Monitor', 'Report', 'Prepare', 'Prevent', 'Resuscitation'])]
        if len(intervention_cols) >= 3:
            labels_df[f"{column_prefix}_Multiple_Interventions"] = (
                labels_df[intervention_cols].sum(axis=1) >= 3
            ).astype(int)
    
    return labels_df

def create_labeled_dataset(csv_file_path, output_csv_path=None, encoding='utf-8-sig'):
    """
    Create a labeled dataset from normalized text columns.
    
    Parameters:
    csv_file_path (str): Path to the input normalized CSV file
    output_csv_path (str, optional): Path to the output labeled CSV file
    encoding (str): Encoding for reading/writing CSV files
    
    Returns:
    str: Path to the created labeled CSV file
    """
    try:
        print(f"Reading normalized dataset: {csv_file_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)
        print(f"Dataset shape: {df.shape}")
        
        # Columns to process
        text_columns = [
            'Nhận định và đánh giá',
            'Kế hoạch (xử trí)'
        ]
        
        # Check if columns exist
        missing_columns = [col for col in text_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            text_columns = [col for col in text_columns if col in df.columns]
        
        print(f"Processing columns: {text_columns}")
        
        # Create labels for each text column
        all_labels = []
        
        for column in text_columns:
            print(f"\nProcessing column: {column}")
            
            # Extract binary labels
            labels_df = extract_labels_from_text(df[column], column)
            print(f"  Created {len(labels_df.columns)} binary labels")
            
            # Create composite labels
            labels_df = create_composite_labels(labels_df, column)
            print(f"  Total labels (including composites): {len(labels_df.columns)}")
            
            all_labels.append(labels_df)
        
        # Combine all labels
        if all_labels:
            combined_labels = pd.concat(all_labels, axis=1)
            print(f"\nTotal label columns created: {len(combined_labels.columns)}")
        else:
            combined_labels = pd.DataFrame(index=df.index)
        
        # Combine original data with labels
        labeled_df = pd.concat([df, combined_labels], axis=1)
        
        print(f"Final dataset shape: {labeled_df.shape}")
        
        # Generate output file path if not provided
        if output_csv_path is None:
            input_path = Path(csv_file_path)
            output_csv_path = input_path.with_stem(f"{input_path.stem}_labeled")
        
        # Save the labeled dataset
        print(f"Saving labeled dataset to: {output_csv_path}")
        
        # Save with UTF-8 BOM for Excel compatibility
        with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            labeled_df.to_csv(f, index=False)
        
        print(f"Successfully created labeled dataset!")
        print(f"Output saved at: {output_csv_path}")
        
        # Create summary report
        create_labeling_report(labeled_df, combined_labels, output_csv_path)
        
        return str(output_csv_path)
        
    except Exception as e:
        print(f"Error creating labeled dataset: {str(e)}")
        return None

def create_labeling_report(full_df, labels_df, output_path):
    """
    Create a summary report of the labeling process.
    
    Parameters:
    full_df (pd.DataFrame): The complete labeled dataframe
    labels_df (pd.DataFrame): Just the label columns
    output_path (str): Path where the main file was saved
    """
    report_path = Path(output_path).with_suffix('.txt').with_stem(f"{Path(output_path).stem}_report")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("LABELING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {Path(output_path).name}\n")
        f.write(f"Total records: {len(full_df):,}\n")
        f.write(f"Original columns: {len(full_df.columns) - len(labels_df.columns)}\n")
        f.write(f"Label columns: {len(labels_df.columns)}\n")
        f.write(f"Total columns: {len(full_df.columns)}\n\n")
        
        # Group labels by category
        assessment_labels = [col for col in labels_df.columns if 'Nhận định và đánh giá' in col]
        treatment_labels = [col for col in labels_df.columns if 'Kế hoạch (xử trí)' in col]
        
        f.write("ASSESSMENT LABELS (Nhận định và đánh giá)\n")
        f.write("-" * 40 + "\n")
        for label in sorted(assessment_labels):
            count = labels_df[label].sum()
            percentage = (count / len(labels_df)) * 100
            f.write(f"{label}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\nTREATMENT PLAN LABELS (Kế hoạch (xử trí))\n")
        f.write("-" * 40 + "\n")
        for label in sorted(treatment_labels):
            count = labels_df[label].sum()
            percentage = (count / len(labels_df)) * 100
            f.write(f"{label}: {count} ({percentage:.1f}%)\n")
        
        # Label correlation analysis
        f.write(f"\nLABEL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Records with any assessment label: {(labels_df[assessment_labels].any(axis=1)).sum()}\n")
        f.write(f"Records with any treatment label: {(labels_df[treatment_labels].any(axis=1)).sum()}\n")
        f.write(f"Records with both types of labels: {(labels_df[assessment_labels].any(axis=1) & labels_df[treatment_labels].any(axis=1)).sum()}\n")
        
        # Most common label combinations
        f.write(f"\nMOST ACTIVE RECORDS\n")
        f.write("-" * 40 + "\n")
        total_labels_per_record = labels_df.sum(axis=1)
        f.write(f"Max labels per record: {total_labels_per_record.max()}\n")
        f.write(f"Average labels per record: {total_labels_per_record.mean():.1f}\n")
        f.write(f"Records with 5+ labels: {(total_labels_per_record >= 5).sum()}\n")
    
    print(f"Labeling report saved to: {report_path}")

def main():
    """
    Main function to create labeled dataset from normalized file.
    """
    # Define paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    data_dir = project_root / "data"
    input_csv = data_dir / "dataset_long_format_normalized.csv"
    
    print("Dataset Labeling: Text to Binary Labels")
    print("=" * 50)
    
    # Check if the input CSV file exists
    if not input_csv.exists():
        print(f"Error: Normalized CSV file not found at {input_csv}")
        return
    
    # Create labeled dataset
    output_file = create_labeled_dataset(str(input_csv))
    
    if output_file:
        print("\n" + "=" * 50)
        print("Labeling completed successfully!")
        print(f"Labeled dataset saved at: {output_file}")
        print("\nBinary labels created for:")
        print("  - Medical assessments (CTG, patient status, complications)")
        print("  - Treatment plans (monitoring, interventions, preparations)")
        print("  - Composite conditions (combinations of factors)")
        print("\nFile is Excel-compatible and ready for ML training!")
    else:
        print("Labeling failed!")

if __name__ == "__main__":
    main()
