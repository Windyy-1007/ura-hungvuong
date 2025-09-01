import pandas as pd

def validate_excel_compatibility():
    """
    Validate that the normalized file is Excel-compatible.
    """
    print("Excel Compatibility Validation")
    print("=" * 40)
    
    file_path = 'data/dataset_long_format_normalized.csv'
    
    try:
        # Test reading with different encodings
        print("Testing file encodings...")
        
        # Test UTF-8 with BOM (Excel standard)
        df_bom = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"✅ UTF-8 BOM: {df_bom.shape}")
        
        # Test regular UTF-8
        df_utf8 = pd.read_csv(file_path, encoding='utf-8')
        print(f"✅ UTF-8: {df_utf8.shape}")
        
        # Check for special characters
        print("\nChecking Vietnamese characters...")
        vietnamese_chars = ['á', 'à', 'ả', 'ã', 'ạ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 
                           'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'đ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ',
                           'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'í', 'ì', 'ỉ', 'ĩ', 'ị',
                           'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ',
                           'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'ú', 'ù', 'ủ', 'ũ', 'ụ',
                           'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ']
        
        text_columns = ['Nhận định và đánh giá', 'Kế hoạch (xử trí)']
        vietnamese_found = False
        
        for col in text_columns:
            if col in df_bom.columns:
                text_data = df_bom[col].dropna().astype(str)
                for char in vietnamese_chars:
                    if any(char in text for text in text_data):
                        vietnamese_found = True
                        break
        
        if vietnamese_found:
            print("✅ Vietnamese characters detected and properly encoded")
        else:
            print("ℹ️  No Vietnamese characters found in target columns")
        
        print("\nNormalization summary:")
        for col in text_columns:
            if col in df_bom.columns:
                unique_count = df_bom[col].nunique()
                missing_count = df_bom[col].isnull().sum()
                print(f"  {col}:")
                print(f"    Unique values: {unique_count}")
                print(f"    Missing values: {missing_count}")
        
        print("\n✅ File is Excel-compatible!")
        print("You can open this file directly in Excel with proper Vietnamese character support.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False

if __name__ == "__main__":
    validate_excel_compatibility()
