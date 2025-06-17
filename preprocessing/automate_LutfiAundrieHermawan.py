import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# -------------------------------
# FUNGSI BANTUAN UNTUK BINNING
# -------------------------------

def bin_age(age):
    if age < 20:
        return 'Teen'
    elif 20 <= age <= 24:
        return 'Young Adult'
    elif 25 <= age <= 29:
        return 'Adult'
    else:
        return 'Older Adult'

def categorize_cgpa(cgpa):
    if cgpa < 2.5:
        return 'Low'
    elif 2.5 <= cgpa <= 3.2:
        return 'Medium'
    else:
        return 'High'

# -------------------------------
# FUNGSI UTAMA PREPROCESSING
# -------------------------------

def preprocess_data(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")
    
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Dataset kosong. Harap periksa file input.")
    print("✅ Dataset berhasil dimuat.")

    # ----------------------------
    # PERBAIKAN TIPE DATA
    # ----------------------------
    cols_to_convert = ['Academic Pressure', 'Work Pressure', 'Financial Stress', 
                       'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'CGPA']
    
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = df[col].replace('?', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------

    # Mengatasi nilai NaN
    # Mengisi NaN dengan 0 sebelum perhitungan
    df['Academic Pressure'] = df['Academic Pressure'].fillna(0)
    df['Work Pressure'] = df['Work Pressure'].fillna(0)
    df['Financial Stress'] = df['Financial Stress'].fillna(0)
    df['Study Satisfaction'] = df['Study Satisfaction'].fillna(1)
    df['Job Satisfaction'] = df['Job Satisfaction'].fillna(1)
    
    # 
    if set(['Academic Pressure', 'Work Pressure', 'Financial Stress']).issubset(df.columns):
        df['Total_Stress'] = (df['Academic Pressure'] + df['Work Pressure'] + df['Financial Stress']).round(3)

    if set(['Study Satisfaction', 'Job Satisfaction']).issubset(df.columns):
        df['Satisfaction_Balance'] = (df['Study Satisfaction'] + df['Job Satisfaction']).round(3)

    if set(['Academic Pressure', 'Work Pressure']).issubset(df.columns):
        df['Pressure_Balance'] = (df['Academic Pressure'] + df['Work Pressure']).round(3)

    if 'Satisfaction_Balance' in df.columns and 'Pressure_Balance' in df.columns:
        df['Stress_Balance_Ratio'] = (df['Pressure_Balance'] / df['Satisfaction_Balance'].replace(0, np.nan)).round(3).fillna(0)

    # ----------------------------
    # BINNING
    # ----------------------------
    if 'Age' in df.columns:
        df['Age_Group'] = df['Age'].apply(bin_age)
    if 'CGPA' in df.columns:
        df['CGPA_Category'] = df['CGPA'].apply(categorize_cgpa)

    # ----------------------------
    # HAPUS KOLOM YANG TIDAK DIPAKAI
    # ----------------------------
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)

    # ----------------------------
    # ONE-HOT ENCODING
    # ----------------------------
    onehot_columns = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    one_hot_encoders = {}
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    encoded_dfs = []
    for col in onehot_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            encoded = encoder.fit_transform(df[[col]])
            feature_names = encoder.get_feature_names_out([col])
            one_hot_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            encoded_dfs.append(one_hot_df)
            one_hot_encoders[col] = encoder
            df = df.drop(col, axis=1)
        else:
            print(f"Kolom '{col}' tidak ditemukan di DataFrame.")

    if encoded_dfs:
        df = pd.concat([df] + encoded_dfs, axis=1)

    # ----------------------------
    # LABEL ENCODING
    # ----------------------------
    label_encoders = {}
    label_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in label_cols:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # ----------------------------
    # SCALING FITUR NUMERIK
    # ----------------------------
    numerical_features = [
        'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction',
        'Work/Study Hours', 'Financial Stress',
        'Total_Stress', 'Satisfaction_Balance', 
        'Pressure_Balance', 'Stress_Balance_Ratio'
    ]
    existing_numerical = [col for col in numerical_features if col in df.columns]
    
    scaler = StandardScaler()
    df[existing_numerical] = scaler.fit_transform(df[existing_numerical])

    # ----------------------------
    # SIMPAN DATASET HASIL
    # ----------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_path, index=False)
    print(f"✅ Dataset berhasil disimpan ke {output_path}")

    return df

# -------------------------------
# EKSEKUSI LANGSUNG
# -------------------------------

if __name__ == "__main__":
    input_path = "student_depression_dataset_raw.csv"
    output_path = "preprocessing/student_depression_preprocessing.csv"
    preprocess_data(input_path, output_path)
