import pandas as pd
import os

csv_path = "heart.csv"  # replace with your file path

# Check if file exists
if os.path.exists(csv_path):
    print("✅ CSV file found!")
else:
    print("❌ CSV file not found!")

# Try loading it
try:
    df = pd.read_csv(csv_path)
    print("✅ CSV loaded successfully!")
    print("First 5 rows:")
    print(df.head())
except Exception as e:
    print("❌ Error loading CSV:", e)
