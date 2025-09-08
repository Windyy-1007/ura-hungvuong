import pandas as pd

def load_data(filename="data/dataset_long_format.csv"):
    """Load dataset từ file CSV"""
    try:
        df = pd.read_csv(filename)
        print(f"✅ Đã load thành công {filename} với {len(df)} dòng, {len(df.columns)} cột.")
        return df
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {filename}")
        return None