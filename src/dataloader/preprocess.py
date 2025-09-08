import pandas as pd
import re

def normalize_targets(df):
    # Nhận định và đánh giá
    df["Nhận định và đánh giá_CTG_Group_I"] = df["Nhận định và đánh giá"].str.contains(r"CTG.*I|CTC.*I", case=False, na=False).astype(int)
    df["Nhận định và đánh giá_CTG_Group_II"] = df["Nhận định và đánh giá"].str.contains(r"CTG.*II|CTC.*II", case=False, na=False).astype(int)
    df["Nhận định và đánh giá_Patient_Stable"] = df["Nhận định và đánh giá"].str.contains(r"sản phụ ổn|ổn định", case=False, na=False).astype(int)
    df["Nhận định và đánh giá_Position_Unfavorable"] = df["Nhận định và đánh giá"].str.contains(r"kiểu thế không thuận|ngôi.*không thuận", case=False, na=False).astype(int)

    # Kế hoạch (xử trí)
    df["Kế hoạch (xử trí)_Monitor_Labor"] = df["Kế hoạch (xử trí)"].str.contains(r"theo dõi|monitor", case=False, na=False).astype(int)
    df["Kế hoạch (xử trí)_Report_Doctor"] = df["Kế hoạch (xử trí)"].str.contains(r"trình bác sĩ|báo.*bác sĩ|bs.*xem|hộ sinh", case=False, na=False).astype(int)
    df["Kế hoạch (xử trí)_Prepare_Delivery"] = df["Kế hoạch (xử trí)"].str.contains(r"chuẩn bị.*sanh", case=False, na=False).astype(int)
    df["Kế hoạch (xử trí)_Any_Resuscitation"] = df["Kế hoạch (xử trí)"].str.contains(r"hồi sức|hsss", case=False, na=False).astype(int)

    return df

# Example usage
df = pd.read_csv("data/dataset_long_format_normalized_labeled.csv")
df = normalize_targets(df)
df.to_csv("data/dataset_long_format_normalized_labeled_targets.csv", index=False)
print("✅ Saved new dataset with standardized targets")