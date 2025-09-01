import pandas as pd
import os
from pathlib import Path

def xlsx_to_csv(xlsx_file_path, csv_file_path=None, sheet_name=None, encoding='utf-8', clean_headers=True):
    """
    Convert XLSX file to CSV with UTF-8 encoding.
    
    Parameters:
    xlsx_file_path (str): Path to the input XLSX file
    csv_file_path (str, optional): Path to the output CSV file. If None, creates CSV with same name as XLSX
    sheet_name (str or int, optional): Sheet name or index to convert. If None, converts the first sheet
    encoding (str): Encoding for the output CSV file (default: utf-8)
    clean_headers (bool): Whether to clean headers by removing newlines and extra spaces (default: True)
    
    Returns:
    str: Path to the created CSV file
    """
    try:
        # Read the XLSX file
        print(f"Reading XLSX file: {xlsx_file_path}")
        
        if sheet_name is not None:
            df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(xlsx_file_path)
        
        # Clean headers if requested
        if clean_headers:
            print("Cleaning column headers...")
            original_columns = df.columns.tolist()
            cleaned_columns = []
            
            for col in original_columns:
                # Convert to string, replace newlines with spaces, and clean up extra spaces
                cleaned_col = str(col).replace('\n', ' ').replace('\r', ' ')
                # Remove multiple consecutive spaces
                cleaned_col = ' '.join(cleaned_col.split())
                cleaned_columns.append(cleaned_col)
            
            df.columns = cleaned_columns
            
            # Report cleaned headers
            newline_cleaned = sum(1 for orig, clean in zip(original_columns, cleaned_columns) if '\n' in str(orig))
            if newline_cleaned > 0:
                print(f"Cleaned {newline_cleaned} column headers that contained newlines")
        
        # Generate CSV file path if not provided
        if csv_file_path is None:
            xlsx_path = Path(xlsx_file_path)
            csv_file_path = xlsx_path.with_suffix('.csv')
        
        # Ensure the output directory exists
        output_dir = Path(csv_file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV with UTF-8 encoding
        print(f"Converting to CSV: {csv_file_path}")
        df.to_csv(csv_file_path, index=False, encoding=encoding)
        
        print(f"Successfully converted {xlsx_file_path} to {csv_file_path}")
        print(f"CSV file created with {len(df)} rows and {len(df.columns)} columns")
        
        return str(csv_file_path)
        
    except FileNotFoundError:
        print(f"Error: File '{xlsx_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        return None

def convert_all_sheets(xlsx_file_path, output_dir=None, encoding='utf-8', clean_headers=True):
    """
    Convert all sheets in an XLSX file to separate CSV files.
    
    Parameters:
    xlsx_file_path (str): Path to the input XLSX file
    output_dir (str, optional): Directory to save CSV files. If None, saves in same directory as XLSX
    encoding (str): Encoding for the output CSV files (default: utf-8)
    clean_headers (bool): Whether to clean headers by removing newlines and extra spaces (default: True)
    
    Returns:
    list: List of paths to created CSV files
    """
    try:
        # Read all sheet names
        xlsx_file = pd.ExcelFile(xlsx_file_path)
        sheet_names = xlsx_file.sheet_names
        
        print(f"Found {len(sheet_names)} sheets: {sheet_names}")
        
        csv_files = []
        
        # Convert each sheet
        for sheet_name in sheet_names:
            df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
            
            # Clean headers if requested
            if clean_headers:
                original_columns = df.columns.tolist()
                cleaned_columns = []
                
                for col in original_columns:
                    # Convert to string, replace newlines with spaces, and clean up extra spaces
                    cleaned_col = str(col).replace('\n', ' ').replace('\r', ' ')
                    # Remove multiple consecutive spaces
                    cleaned_col = ' '.join(cleaned_col.split())
                    cleaned_columns.append(cleaned_col)
                
                df.columns = cleaned_columns
                
                # Report cleaned headers for this sheet
                newline_cleaned = sum(1 for orig, clean in zip(original_columns, cleaned_columns) if '\n' in str(orig))
                if newline_cleaned > 0:
                    print(f"  - Cleaned {newline_cleaned} column headers in sheet '{sheet_name}'")
            
            # Generate CSV file path
            if output_dir is None:
                output_dir = Path(xlsx_file_path).parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            xlsx_stem = Path(xlsx_file_path).stem
            csv_file_path = output_dir / f"{xlsx_stem}_{sheet_name}.csv"
            
            # Save as CSV
            df.to_csv(csv_file_path, index=False, encoding=encoding)
            csv_files.append(str(csv_file_path))
            
            print(f"Converted sheet '{sheet_name}' to {csv_file_path}")
            print(f"  - {len(df)} rows and {len(df.columns)} columns")
        
        return csv_files
        
    except Exception as e:
        print(f"Error converting sheets: {str(e)}")
        return []

def main():
    """
    Main function to demonstrate the conversion.
    Converts the dataset.xlsx file in the data directory.
    """
    # Define paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / "data"
    xlsx_file = data_dir / "dataset.xlsx"
    
    print("XLSX to CSV Converter")
    print("=" * 40)
    
    # Check if the XLSX file exists
    if not xlsx_file.exists():
        print(f"Error: XLSX file not found at {xlsx_file}")
        return
    
    # Convert the main sheet to CSV
    csv_file = xlsx_to_csv(str(xlsx_file))
    
    if csv_file:
        print(f"\nConversion completed successfully!")
        print(f"CSV file saved at: {csv_file}")
        
        # Optionally, also convert all sheets to separate CSV files
        print("\nConverting all sheets to separate CSV files...")
        csv_files = convert_all_sheets(str(xlsx_file), str(data_dir))
        
        if csv_files:
            print(f"Created {len(csv_files)} CSV files:")
            for csv_file in csv_files:
                print(f"  - {csv_file}")
    else:
        print("Conversion failed!")

def convert_xlsx_file(input_path, output_path=None):
    """
    Simple utility function to convert a single XLSX file to CSV.
    
    Parameters:
    input_path (str): Path to the input XLSX file
    output_path (str, optional): Path to the output CSV file
    
    Returns:
    str: Path to the created CSV file
    """
    return xlsx_to_csv(input_path, output_path, encoding='utf-8')

if __name__ == "__main__":
    main()
