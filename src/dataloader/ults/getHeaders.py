import pandas as pd
import os
from pathlib import Path

def get_csv_headers(csv_file_path, output_txt_path=None, encoding='utf-8'):
    """
    Extract headers from a CSV file and export them to a text file.
    
    Parameters:
    csv_file_path (str): Path to the input CSV file
    output_txt_path (str, optional): Path to the output text file. If None, creates TXT with same name as CSV
    encoding (str): Encoding for reading the CSV file (default: utf-8)
    
    Returns:
    tuple: (list of headers, path to output text file)
    """
    try:
        # Read only the first row (headers) from CSV
        print(f"Reading headers from CSV file: {csv_file_path}")
        
        # Read just the header row to be efficient
        df = pd.read_csv(csv_file_path, nrows=0, encoding=encoding)
        headers = df.columns.tolist()
        
        print(f"Found {len(headers)} column headers")
        
        # Generate output text file path if not provided
        if output_txt_path is None:
            csv_path = Path(csv_file_path)
            output_txt_path = csv_path.with_suffix('.txt').with_stem(f"{csv_path.stem}_headers")
        
        # Ensure the output directory exists
        output_dir = Path(output_txt_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write headers to text file, one header per line
        print(f"Writing headers to text file: {output_txt_path}")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for i, header in enumerate(headers, 1):
                f.write(f"{header}\n")
        
        print(f"Successfully exported {len(headers)} headers to {output_txt_path}")
        
        # Display first few headers as preview
        print("\nPreview of headers:")
        for i, header in enumerate(headers[:10], 1):
            print(f"  {i:3d}. {header}")
        
        if len(headers) > 10:
            print(f"  ... and {len(headers) - 10} more headers")
        
        return headers, str(output_txt_path)
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None, None

def get_headers_with_index(csv_file_path, output_txt_path=None, encoding='utf-8'):
    """
    Extract headers from a CSV file with index numbers and export them to a text file.
    
    Parameters:
    csv_file_path (str): Path to the input CSV file
    output_txt_path (str, optional): Path to the output text file with indexed headers
    encoding (str): Encoding for reading the CSV file (default: utf-8)
    
    Returns:
    tuple: (list of headers, path to output text file)
    """
    try:
        # Read only the first row (headers) from CSV
        print(f"Reading headers from CSV file: {csv_file_path}")
        
        # Read just the header row to be efficient
        df = pd.read_csv(csv_file_path, nrows=0, encoding=encoding)
        headers = df.columns.tolist()
        
        print(f"Found {len(headers)} column headers")
        
        # Generate output text file path if not provided
        if output_txt_path is None:
            csv_path = Path(csv_file_path)
            output_txt_path = csv_path.with_suffix('.txt').with_stem(f"{csv_path.stem}_headers_indexed")
        
        # Ensure the output directory exists
        output_dir = Path(output_txt_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write headers to text file with index numbers
        print(f"Writing indexed headers to text file: {output_txt_path}")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for i, header in enumerate(headers, 1):
                f.write(f"{i:4d}. {header}\n")
        
        print(f"Successfully exported {len(headers)} indexed headers to {output_txt_path}")
        
        return headers, str(output_txt_path)
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None, None

def search_headers(csv_file_path, search_term, encoding='utf-8'):
    """
    Search for headers containing a specific term.
    
    Parameters:
    csv_file_path (str): Path to the input CSV file
    search_term (str): Term to search for in header names
    encoding (str): Encoding for reading the CSV file (default: utf-8)
    
    Returns:
    list: List of matching headers with their indices
    """
    try:
        df = pd.read_csv(csv_file_path, nrows=0, encoding=encoding)
        headers = df.columns.tolist()
        
        # Search for headers containing the search term (case-insensitive)
        matching_headers = []
        for i, header in enumerate(headers, 1):
            if search_term.lower() in header.lower():
                matching_headers.append((i, header))
        
        print(f"Found {len(matching_headers)} headers containing '{search_term}':")
        for index, header in matching_headers:
            print(f"  {index:4d}. {header}")
        
        return matching_headers
        
    except Exception as e:
        print(f"Error searching headers: {str(e)}")
        return []

def main():
    """
    Main function to extract headers from the dataset CSV file.
    """
    # Define paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    data_dir = project_root / "data"
    csv_file = data_dir / "dataset.csv"
    
    print("CSV Headers Extractor")
    print("=" * 40)
    
    # Check if the CSV file exists
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return
    
    # Extract headers and save to text file
    headers, output_file = get_csv_headers(str(csv_file))
    
    if headers:
        print(f"\nHeaders successfully extracted!")
        print(f"Text file saved at: {output_file}")
        
        # Also create an indexed version
        print("\nCreating indexed version...")
        indexed_headers, indexed_output_file = get_headers_with_index(str(csv_file))
        
        if indexed_headers:
            print(f"Indexed headers file saved at: {indexed_output_file}")
        
        # Example search functionality
        print("\n" + "=" * 40)
        print("Example: Searching for headers containing 'Ngày'")
        search_headers(str(csv_file), "Ngày")
        
        print("\nExample: Searching for headers containing 'Apgar'")
        search_headers(str(csv_file), "Apgar")
    else:
        print("Failed to extract headers!")

def extract_headers_to_txt(csv_file_path, output_txt_path=None):
    """
    Simple utility function to extract headers from CSV and save to text file.
    
    Parameters:
    csv_file_path (str): Path to the input CSV file
    output_txt_path (str, optional): Path to the output text file
    
    Returns:
    str: Path to the created text file
    """
    return get_csv_headers(csv_file_path, output_txt_path)[1]

if __name__ == "__main__":
    main()
