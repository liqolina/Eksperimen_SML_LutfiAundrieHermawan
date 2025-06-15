import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

def preprocess_data(input_path, output_path):
    # Cek file input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")
    
    # Memuat dataset
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Dataset kosong. Harap periksa file input.")
    print("Dataset berhasil dimuat.")

    # -------------------------------
    # Encoding Data Kategorikal
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    label_encoders = {}

    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')  # Tangani nilai null
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # -------------------------------
    # Normalisasi / Standarisasi Fitur Numerik
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # -------------------------------
    # Simpan hasil ke output_path
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_path, index=False)
    print(f"Dataset berhasil disimpan ke {output_path}")

    return df

if __name__ == "__main__":
    input_path = 'student_depression_dataset_raw.csv'
    output_path = 'preprocessing/student_depression_preprocessing.csv'
    preprocess_data(input_path, output_path)
