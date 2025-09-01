import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
import unicodedata

def normalize_text_column(series, column_name):
    """
    Normalize text values in a pandas Series.
    
    Parameters:
    series (pd.Series): The series to normalize
    column_name (str): Name of the column for logging
    
    Returns:
    pd.Series: Normalized series
    """
    print(f"\nNormalizing column: {column_name}")
    
    # Create a copy to work with
    normalized = series.copy()
    
    # Step 1: Handle missing values
    initial_missing = normalized.isnull().sum()
    print(f"  Initial missing values: {initial_missing}")
    
    # Step 2: Convert to string and handle various null representations
    normalized = normalized.astype(str)
    normalized = normalized.replace(['nan', 'NaN', 'None', 'null', ''], np.nan)
    
    # Step 3: Basic text cleaning
    def clean_text(text):
        if pd.isna(text) or text == 'nan':
            return np.nan
        
        # Convert to string
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove leading/trailing punctuation if standalone
        if text in ['.', ',', ';', ':', '-', '_']:
            return np.nan
            
        return text if text else np.nan
    
    normalized = normalized.apply(clean_text)
    
    # Step 4: Create normalization mapping based on similarity
    unique_values = normalized.dropna().unique()
    print(f"  Unique values before normalization: {len(unique_values)}")
    
    # Create mapping dictionary for common variations
    normalization_map = create_normalization_mapping(unique_values, column_name)
    
    # Apply normalization mapping
    normalized = normalized.map(lambda x: normalization_map.get(x, x) if pd.notna(x) else x)
    
    final_unique = normalized.dropna().unique()
    print(f"  Unique values after normalization: {len(final_unique)}")
    
    # Show top values after normalization
    value_counts = normalized.value_counts().head(10)
    print(f"  Top normalized values:")
    for value, count in value_counts.items():
        print(f"    {value}: {count}")
    
    return normalized

def create_normalization_mapping(unique_values, column_name):
    """
    Create a mapping for normalizing similar text values.
    
    Parameters:
    unique_values (array): Array of unique values
    column_name (str): Name of the column for specific rules
    
    Returns:
    dict: Mapping dictionary
    """
    mapping = {}
    
    if column_name == "Nhận định và đánh giá":
        # Define normalization rules for "Nhận định và đánh giá"
        ctg_patterns = {
            'CTG nhóm I': ['ctg nhóm i', 'ctg i', 'ctg group i', 'ctg 1'],
            'CTG nhóm II': ['ctg nhóm ii', 'ctg ii', 'ctg group ii', 'ctg 2', 'ctg nhóm 2'],
            'CTG nhóm III': ['ctg nhóm iii', 'ctg iii', 'ctg group iii', 'ctg 3'],
            'Hướng dẫn rặn': ['huong dan ran', 'hướng dẫn rặn sanh', 'hd rặn'],
            'Sản phụ ổn': ['san phu on', 'sp ổn', 'bệnh nhân ổn'],
            'Kiểu thế không thuận lợi': ['kiểu thế không thuận lợi', 'kt không thuận lợi', 'kieu the khong thuan loi'],
            'Đa sản': ['da san', 'đa sanh'],
        }
        
        # Apply pattern matching
        for standard, patterns in ctg_patterns.items():
            for value in unique_values:
                value_lower = str(value).lower().strip()
                if any(pattern in value_lower for pattern in patterns):
                    mapping[value] = standard
    
    elif column_name == "Kế hoạch (xử trí)":
        # Define normalization rules for "Kế hoạch (xử trí)"
        plan_patterns = {
            'Theo dõi tiếp chuyển dạ': [
                'theo doi tiep chuyen da', 'theo dõi tiếp cd', 'td tiếp cd',
                'theo dõi chuyển dạ', 'follow up labor'
            ],
            'Trình bác sĩ': [
                'trinh bac si', 'trình bs', 'báo cáo bác sĩ', 'report doctor'
            ],
            'Chuẩn bị dụng cụ sanh + HSSS': [
                'chuan bi dung cu sanh', 'cb dc sanh', 'chuẩn bị đỡ đẻ',
                'prepare delivery', 'hsss', 'hồi sức sơ sinh'
            ],
            'Đề phòng băng huyết sau sinh': [
                'de phong bang huyet', 'dpbh', 'đề phòng băng huyết',
                'prevent postpartum hemorrhage', 'ppv băng huyết'
            ],
            'Hồi sức thai': [
                'hoi suc thai', 'resuscitation', 'hồi sức thai nhi'
            ]
        }
        
        # Apply pattern matching
        for standard, patterns in plan_patterns.items():
            for value in unique_values:
                value_lower = str(value).lower().strip()
                if any(pattern in value_lower for pattern in patterns):
                    mapping[value] = standard
    
    # Handle compound values (separated by commas)
    for value in unique_values:
        if ', ' in str(value):
            parts = [part.strip() for part in str(value).split(',')]
            normalized_parts = []
            for part in parts:
                # Try to find a normalized version of each part
                normalized_part = mapping.get(part, part)
                if normalized_part not in normalized_parts:
                    normalized_parts.append(normalized_part)
            
            if len(normalized_parts) > 1:
                mapping[value] = ', '.join(sorted(normalized_parts))
    
    return mapping

def normalize_dataset(csv_file_path, output_csv_path=None, encoding='utf-8'):
    """
    Normalize the specified text columns in the dataset and ensure Excel compatibility.
    
    Parameters:
    csv_file_path (str): Path to the input CSV file
    output_csv_path (str, optional): Path to the output CSV file
    encoding (str): Encoding for reading/writing CSV files (default: utf-8)
    
    Returns:
    str: Path to the normalized CSV file
    """
    try:
        print(f"Reading dataset: {csv_file_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)
        print(f"Dataset shape: {df.shape}")
        
        # Columns to normalize
        columns_to_normalize = [
            'Nhận định và đánh giá',
            'Kế hoạch (xử trí)'
        ]
        
        # Check if columns exist
        missing_columns = [col for col in columns_to_normalize if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            columns_to_normalize = [col for col in columns_to_normalize if col in df.columns]
        
        print(f"Normalizing columns: {columns_to_normalize}")
        
        # Normalize each specified column
        for column in columns_to_normalize:
            df[column] = normalize_text_column(df[column], column)
        
        # Generate output file path if not provided
        if output_csv_path is None:
            input_path = Path(csv_file_path)
            output_csv_path = input_path.with_stem(f"{input_path.stem}_normalized")
        
        # Ensure the output directory exists
        output_dir = Path(output_csv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with UTF-8 BOM for Excel compatibility
        print(f"Saving normalized dataset to: {output_csv_path}")
        
        # Save with UTF-8 BOM (Excel-compatible)
        with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            df.to_csv(f, index=False)
        
        print(f"Successfully normalized dataset!")
        print(f"Output saved at: {output_csv_path}")
        
        # Create summary report
        create_normalization_report(df, columns_to_normalize, output_csv_path)
        
        return str(output_csv_path)
        
    except Exception as e:
        print(f"Error normalizing dataset: {str(e)}")
        return None

def create_normalization_report(df, normalized_columns, output_path):
    """
    Create a summary report of the normalization process.
    
    Parameters:
    df (pd.DataFrame): The normalized dataframe
    normalized_columns (list): List of columns that were normalized
    output_path (str): Path where the main file was saved
    """
    report_path = Path(output_path).with_suffix('.txt').with_stem(f"{Path(output_path).stem}_report")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("NORMALIZATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {Path(output_path).name}\n")
        f.write(f"Total records: {len(df):,}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Normalized columns: {len(normalized_columns)}\n\n")
        
        for column in normalized_columns:
            f.write(f"COLUMN: {column}\n")
            f.write("-" * 40 + "\n")
            
            # Value counts
            value_counts = df[column].value_counts()
            f.write(f"Unique values: {len(value_counts)}\n")
            f.write(f"Missing values: {df[column].isnull().sum()}\n")
            f.write(f"Most common values:\n")
            
            for value, count in value_counts.head(15).items():
                percentage = (count / len(df)) * 100
                f.write(f"  {value}: {count} ({percentage:.1f}%)\n")
            
            f.write("\n")
    
    print(f"Normalization report saved to: {report_path}")

def main():
    """
    Main function to normalize the long format dataset.
    """
    # Define paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    data_dir = project_root / "data"
    input_csv = data_dir / "dataset_long_format.csv"
    
    print("Dataset Text Normalization")
    print("=" * 50)
    
    # Check if the input CSV file exists
    if not input_csv.exists():
        print(f"Error: CSV file not found at {input_csv}")
        return
    
    # Normalize the dataset
    output_file = normalize_dataset(str(input_csv))
    
    if output_file:
        print("\n" + "=" * 50)
        print("Normalization completed successfully!")
        print(f"Normalized data saved at: {output_file}")
        print("\nFile is Excel-compatible with UTF-8 BOM encoding")
        print("You can open it directly in Excel with proper character support")
    else:
        print("Normalization failed!")

if __name__ == "__main__":
    main()
