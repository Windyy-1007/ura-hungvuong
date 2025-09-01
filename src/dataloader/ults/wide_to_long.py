import pandas as pd
import numpy as np
from pathlib import Path

def reshape_wide_to_long(csv_file_path, output_csv_path=None, encoding='utf-8'):
    """
    Transform the dataset from wide format (general, hour_0, hour_1, ..., hour_29) 
    to long format (multiple rows per patient, each with general + one hour's data).
    
    Parameters:
    csv_file_path (str): Path to the input wide-format CSV file
    output_csv_path (str, optional): Path to the output long-format CSV file
    encoding (str): Encoding for reading/writing CSV files (default: utf-8)
    
    Returns:
    str: Path to the created long-format CSV file
    """
    try:
        print(f"Reading wide-format CSV file: {csv_file_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)
        print(f"Original data shape: {df.shape}")
        
        # Define the general information columns (first 12 columns, excluding "Số cột")
        general_columns = [
            'Dấu thời gian',
            'Mã bệnh nhân', 
            'Họ và tên',
            'Năm sinh',
            'Para (điền 4 số)',
            'Tiền căn bệnh lý',
            'Khởi phát chuyển dạ (1: Có, 0: Không)',
            'Chẩn đoán chuyển dạ hoạt động',
            'Ngày ối vỡ',
            'Giờ ối vỡ',
            'Yếu tố nguy cơ',
            'Ngày chuyển dạ hoạt động (xx/yy/zzzz)'
        ]
        
        # Skip "Số cột" column (index 12)
        skip_column = 'Số cột'
        
        # Define the time series feature names (34 features per time point)
        timeseries_features = [
            'Ngày',
            'Giờ',
            'Bạn đồng hành (1: Có, 0: Không)',
            'Đánh giá mức độ đau (VAS) (Điền số nguyên)',
            'Nước uống vào (1: Có, 0: Không)',
            'Ăn',
            'Mạch (nhập số nguyên)',
            'HA tâm thu (nhập số nguyên)',
            'HA tâm trương (nhập số nguyên)',
            'Nhiệt độ (nhập số nguyên)',
            'Nước tiểu',
            'TT cơ bản (nhập số nguyên)',
            'CTG',
            'Nước ối (V: ối vỡ/Vg: Vàng)',
            'Kiểu thế',
            'Bướu HT',
            'Chồng khớp',
            'Các cơn co TC/10 phút (điền số nguyên)',
            'Thời gian của các cơn co TC (điền số nguyên)',
            'Cổ TC (KH: X)',
            'Nếu 10: 10X hay 10R? (Không phải 10 xin bỏ qua)',
            'Độ lọt (KH: O)',
            'Oxytocin (số hoặc số la mã)',
            'Thuốc',
            'Thuốc gì?',
            'Truyền dịch',
            'Nhận định và đánh giá',
            'Kế hoạch (xử trí)',
            'Sanh',
            'Giờ sanh (chỉ điền khi không phải sanh mổ)',
            'Lý do mổ (nếu có):',
            'Giới tính em bé',
            'Apgar (nhập phân số số nguyên X/Y) (ví dụ: 1/5)',
            'Cân nặng'
        ]
        
        print(f"General columns: {len(general_columns)}")
        print(f"Time series features per hour: {len(timeseries_features)}")
        
        # Extract general information
        general_data = df[general_columns].copy()
        print(f"General data shape: {general_data.shape}")
        
        # Prepare list to store long format data
        long_format_rows = []
        
        # Process each time point (0 to 29)
        for hour in range(30):
            print(f"Processing hour {hour}...")
            
            # Create column names for this time point
            if hour == 0:
                # First time point has no suffix
                hour_columns = timeseries_features.copy()
            else:
                # Subsequent time points have .{hour} suffix
                hour_columns = [f"{feature}.{hour}" for feature in timeseries_features]
            
            # Check if all columns exist in the dataset
            existing_columns = [col for col in hour_columns if col in df.columns]
            missing_columns = [col for col in hour_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  Warning: Missing columns for hour {hour}: {len(missing_columns)} columns")
                # Skip this hour if too many columns are missing
                if len(missing_columns) > len(hour_columns) // 2:
                    print(f"  Skipping hour {hour} due to too many missing columns")
                    continue
            
            # Extract time series data for this hour
            hour_data = df[existing_columns].copy()
            
            # Rename columns to remove the suffix
            if hour > 0:
                column_mapping = {}
                for old_col in existing_columns:
                    if old_col.endswith(f'.{hour}'):
                        new_col = old_col.replace(f'.{hour}', '')
                        column_mapping[old_col] = new_col
                hour_data = hour_data.rename(columns=column_mapping)
            
            # Add missing columns with NaN values
            for feature in timeseries_features:
                if feature not in hour_data.columns:
                    hour_data[feature] = np.nan
            
            # Reorder columns to match the original order
            hour_data = hour_data[timeseries_features]
            
            # Add hour identifier
            hour_data['hour'] = hour
            
            # Combine general info with this hour's data
            combined_data = pd.concat([general_data.reset_index(drop=True), 
                                     hour_data.reset_index(drop=True)], axis=1)
            
            # Filter out completely empty rows (where all time series features are NaN)
            non_empty_mask = combined_data[timeseries_features].notna().any(axis=1)
            combined_data = combined_data[non_empty_mask]
            
            print(f"  Hour {hour}: {combined_data.shape[0]} non-empty records")
            
            long_format_rows.append(combined_data)
        
        # Combine all time points
        print("Combining all time points...")
        long_format_df = pd.concat(long_format_rows, ignore_index=True)
        
        # Reorder columns: general columns, hour, then time series features
        final_columns = general_columns + ['hour'] + timeseries_features
        long_format_df = long_format_df[final_columns]
        
        print(f"Long format data shape: {long_format_df.shape}")
        print(f"Patients in original data: {len(df)}")
        print(f"Total records in long format: {len(long_format_df)}")
        print(f"Average records per patient: {len(long_format_df) / len(df):.1f}")
        
        # Generate output file path if not provided
        if output_csv_path is None:
            input_path = Path(csv_file_path)
            output_csv_path = input_path.with_stem(f"{input_path.stem}_long_format")
        
        # Save the long format data
        print(f"Saving long format data to: {output_csv_path}")
        long_format_df.to_csv(output_csv_path, index=False, encoding=encoding)
        
        print(f"Successfully transformed data to long format!")
        print(f"Output saved at: {output_csv_path}")
        
        return str(output_csv_path)
        
    except Exception as e:
        print(f"Error transforming data: {str(e)}")
        return None

def analyze_long_format_data(long_csv_path, encoding='utf-8'):
    """
    Analyze the structure and patterns in the long format data.
    
    Parameters:
    long_csv_path (str): Path to the long-format CSV file
    encoding (str): Encoding for reading CSV file (default: utf-8)
    """
    try:
        print("Analyzing long format data...")
        df = pd.read_csv(long_csv_path, encoding=encoding)
        
        print(f"Total records: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        # Analyze by patient
        patient_counts = df.groupby('Mã bệnh nhân').size()
        print(f"\nRecords per patient:")
        print(f"  Mean: {patient_counts.mean():.1f}")
        print(f"  Median: {patient_counts.median():.1f}")
        print(f"  Min: {patient_counts.min()}")
        print(f"  Max: {patient_counts.max()}")
        
        # Analyze by hour
        hour_counts = df['hour'].value_counts().sort_index()
        print(f"\nRecords per hour:")
        for hour in range(min(10, len(hour_counts))):
            if hour in hour_counts.index:
                print(f"  Hour {hour}: {hour_counts[hour]} records")
        if len(hour_counts) > 10:
            print(f"  ... and {len(hour_counts) - 10} more hours")
        
        # Check for missing data patterns
        print(f"\nMissing data analysis:")
        missing_percentages = (df.isnull().sum() / len(df) * 100).round(1)
        high_missing = missing_percentages[missing_percentages > 50]
        if len(high_missing) > 0:
            print(f"  Columns with >50% missing data: {len(high_missing)}")
            for col, pct in high_missing.head().items():
                print(f"    {col}: {pct}%")
        
        return df
        
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return None

def main():
    """
    Main function to transform the dataset from wide to long format.
    """
    # Define paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    input_csv = data_dir / "dataset.csv"
    
    print("Dataset Transformation: Wide to Long Format")
    print("=" * 50)
    
    # Check if the input CSV file exists
    if not input_csv.exists():
        print(f"Error: CSV file not found at {input_csv}")
        return
    
    # Transform the data
    output_file = reshape_wide_to_long(str(input_csv))
    
    if output_file:
        print("\n" + "=" * 50)
        # Analyze the transformed data
        analyze_long_format_data(output_file)
        
        print(f"\nTransformation completed successfully!")
        print(f"Long format data saved at: {output_file}")
    else:
        print("Transformation failed!")

if __name__ == "__main__":
    main()
