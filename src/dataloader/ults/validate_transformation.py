import pandas as pd

# Read the long format data
df = pd.read_csv('data/dataset_long_format.csv')

print("Long Format Data Analysis")
print("=" * 40)
print(f"Shape: {df.shape}")
print(f"Total patients: {df['Mã bệnh nhân'].nunique()}")
print(f"Total records: {len(df)}")
print(f"Average records per patient: {len(df) / df['Mã bệnh nhân'].nunique():.1f}")

print("\nColumn structure:")
print(f"General columns: 12 (patient info)")
print(f"Hour column: 1")
print(f"Time series columns: 34 (hourly measurements)")
print(f"Total columns: {len(df.columns)}")

print("\nFirst patient example:")
first_patient_id = df['Mã bệnh nhân'].iloc[0]
first_patient = df[df['Mã bệnh nhân'] == first_patient_id]
print(f"Patient {first_patient_id} has {len(first_patient)} records")

print("\nTime points (hours) for first patient:")
print(first_patient[['hour', 'Ngày', 'Giờ', 'Mạch (nhập số nguyên)', 'HA tâm thu (nhập số nguyên)']].to_string(index=False))

print("\nHour distribution across all patients:")
hour_dist = df['hour'].value_counts().sort_index()
for hour in hour_dist.index[:10]:
    print(f"  Hour {hour}: {hour_dist[hour]} records")
if len(hour_dist) > 10:
    print(f"  ... and {len(hour_dist) - 10} more hours")
