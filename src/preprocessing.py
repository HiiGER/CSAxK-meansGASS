import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_raw_data(filepath='data/Data Set UMKM.xlsx'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    absolute_filepath = os.path.join(project_root, filepath)
    return pd.read_excel(absolute_filepath, header=1)

def extract_lama_usaha(df):
    def extract_year(date_str):
        if pd.isna(date_str) or str(date_str).strip() == '-' or str(date_str).strip() == '':
            return np.nan
        match = re.search(r'\d{4}', str(date_str))
        if match:
            return int(match.group(0))
        return np.nan

    df['Tahun Pendirian'] = df['Tanggal Pendirian Usaha'].apply(extract_year)
    current_year = datetime.now().year
    df['Lama Usaha (Tahun)'] = current_year - df['Tahun Pendirian']
    df['Lama Usaha (Tahun)'] = df['Lama Usaha (Tahun)'].fillna(df['Lama Usaha (Tahun)'].median())
    return df

def to_binary(val):
    if pd.isna(val) or str(val).strip() == '-' or str(val).strip() == '' or str(val).lower() == 'tidak':
        return 0
    return 1

def map_omset(val):
    omset_mapping = {'Kurang dari 10 juta': 1, '10 - 50 Juta': 2, '50 - 300 Juta': 3, 'Lebih dari 300 Juta': 4}
    val_str = str(val).strip().lower()
    for key, num in omset_mapping.items():
        if key.lower() in val_str:
            return num
    return 1

def scale_features(df_final, features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_final)
    return pd.DataFrame(scaled_data, columns=features)

# =======================================================
# ORIGINAL FUNCTION (For backward compatibility)
# =======================================================
def clean_and_preprocess_data(filepath='data/Data Set UMKM.xlsx'):
    return preprocess_finansial(filepath)

# =======================================================
# CASE 1: KAPASITAS FINANSIAL & SKALA BISNIS
# =======================================================
def preprocess_finansial(filepath='data/Data Set UMKM.xlsx'):
    df = load_raw_data(filepath)
    df = extract_lama_usaha(df)
    
    df['Menerima Bantuan'] = df['Modal Bantuan Pemerintah'].apply(to_binary)
    df['Punya Pinjaman KUR'] = df['Pinjaman Kredit Usaha Rakyat'].apply(to_binary)
    df['Tingkat Omset'] = df['Omset per-Tahun'].apply(map_omset)
    df['Laki-laki'] = pd.to_numeric(df['Laki-laki'].replace('-', 0), errors='coerce').fillna(0)
    df['Perempuan'] = pd.to_numeric(df['Perempuan'].replace('-', 0), errors='coerce').fillna(0)
    df['Total Karyawan'] = df['Laki-laki'] + df['Perempuan']
    
    features = ['Lama Usaha (Tahun)', 'Menerima Bantuan', 'Punya Pinjaman KUR', 'Tingkat Omset', 'Total Karyawan']
    df_final = df[features].copy()
    df_scaled = scale_features(df_final, features)
    return df_final, df_scaled, features

# =======================================================
# CASE 2: DEMOGRAFI PEMILIK VS KETAHANAN USAHA
# =======================================================
def preprocess_demografi(filepath='data/Data Set UMKM.xlsx'):
    df = load_raw_data(filepath)
    df = extract_lama_usaha(df)
    
    # Clean Usia
    df['Usia Pemilik'] = pd.to_numeric(df['Usia'], errors='coerce').fillna(df['Usia'].median() if pd.api.types.is_numeric_dtype(df['Usia']) else 40)
    
    # Encode Jenis Kelamin
    df['Is_Laki'] = df['Jenis Kelamin'].apply(lambda x: 1 if str(x).strip().upper() == 'L' else 0)
    
    # Encode Pendidikan Terakhir
    edu_map = {'SD': 1, 'SMP': 2, 'SMA': 3, 'D3': 4, 'S1': 5, 'S2': 6, 'S3': 7}
    def map_edu(val):
        for k, v in edu_map.items():
            if k in str(val).upper(): return v
        return 2 # Default to SMP if unknown
    df['Tingkat Pendidikan'] = df['Pendidikan Terakhir'].apply(map_edu)
    
    features = ['Usia Pemilik', 'Is_Laki', 'Tingkat Pendidikan', 'Lama Usaha (Tahun)']
    df_final = df[features].copy()
    
    # Fill any remaining NaNs with column median
    df_final = df_final.fillna(df_final.median())
    
    df_scaled = scale_features(df_final, features)
    return df_final, df_scaled, features

# =======================================================
# CASE 3: DIGITALISASI & JANGKAUAN PEMASARAN
# =======================================================
def preprocess_digitalisasi(filepath='data/Data Set UMKM.xlsx'):
    df = load_raw_data(filepath)
    
    # Score Social Media (count commas + 1 if not empty)
    def count_medsos(val):
        if pd.isna(val) or str(val).strip() == '-': return 0
        return len(str(val).split(','))
    df['Skor Digitalisasi'] = df['Sarana Media Elektronik'].apply(count_medsos)
    
    df['Ekspor'] = df['Produk Komoditas Ekspor'].apply(lambda x: 1 if str(x).strip().lower() == 'ya' else 0)
    
    def map_pemasaran(val):
        val_str = str(val).lower()
        if 'luar negeri' in val_str: return 4
        elif 'nasional' in val_str or 'luar pulau' in val_str or 'pulau jawa' in val_str: return 3
        elif 'diy' in val_str: return 2
        return 1 # Lokal kota/kab
    df['Jangkauan Pasar'] = df['Tujuan Pemasaran'].apply(map_pemasaran)
    
    df['Tingkat Omset'] = df['Omset per-Tahun'].apply(map_omset)
    
    features = ['Skor Digitalisasi', 'Ekspor', 'Jangkauan Pasar', 'Tingkat Omset']
    df_final = df[features].copy()
    df_final = df_final.fillna(df_final.median())
    df_scaled = scale_features(df_final, features)
    return df_final, df_scaled, features

# =======================================================
# CASE 4: KEMANDIRIAN ASET & PEMODALAN
# =======================================================
def preprocess_aset_modal(filepath='data/Data Set UMKM.xlsx'):
    df = load_raw_data(filepath)
    
    # Note the typo in the original dataset 'Kepemilkan' instead of 'Kepemilikan'
    df['Milik Sendiri'] = df['Status Kepemilkan Tanah/Bangunan'].apply(lambda x: 1 if 'milik sendiri' in str(x).lower() else 0)
    df['Punya Pinjaman KUR'] = df['Pinjaman Kredit Usaha Rakyat'].apply(to_binary)
    df['Menerima Bantuan'] = df['Modal Bantuan Pemerintah'].apply(to_binary)
    df['Tingkat Omset'] = df['Omset per-Tahun'].apply(map_omset)
    
    features = ['Milik Sendiri', 'Punya Pinjaman KUR', 'Menerima Bantuan', 'Tingkat Omset']
    df_final = df[features].copy()
    df_final = df_final.fillna(df_final.median())
    df_scaled = scale_features(df_final, features)
    return df_final, df_scaled, features

# =======================================================
# CASE 5: PROFIL KETENAGAKERJAAN
# =======================================================
def preprocess_ketenagakerjaan(filepath='data/Data Set UMKM.xlsx'):
    df = load_raw_data(filepath)
    
    df['Laki-laki'] = pd.to_numeric(df['Laki-laki'].replace('-', 0), errors='coerce').fillna(0)
    df['Perempuan'] = pd.to_numeric(df['Perempuan'].replace('-', 0), errors='coerce').fillna(0)
    df['Total Karyawan'] = df['Laki-laki'] + df['Perempuan']
    
    # Rasio Perempuan
    df['Rasio Perempuan'] = np.where(df['Total Karyawan'] > 0, df['Perempuan'] / df['Total Karyawan'], 0)
    
    def map_usia_pekerja(val):
        val_str = str(val).lower()
        if '17-25' in val_str: return 1
        elif '26-35' in val_str: return 2
        elif '35-50' in val_str: return 3
        elif 'lebih dari 50' in val_str: return 4
        return 2 # default
    df['Rentang Usia Pekerja'] = df['Rerata Usia Pekerja'].apply(map_usia_pekerja)
    
    df['Punya BPJS/Asuransi'] = df['Kepemilikan Asuransi Kesehatan'].apply(lambda x: 1 if 'bpjs' in str(x).lower() or 'asuransi' in str(x).lower() else 0)
    
    features = ['Total Karyawan', 'Rasio Perempuan', 'Rentang Usia Pekerja', 'Punya BPJS/Asuransi']
    df_final = df[features].copy()
    df_final = df_final.fillna(df_final.median())
    df_scaled = scale_features(df_final, features)
    return df_final, df_scaled, features

if __name__ == "__main__":
    print("Testing Preprocessing Modules...")
    funcs = [preprocess_finansial, preprocess_demografi, preprocess_digitalisasi, preprocess_aset_modal, preprocess_ketenagakerjaan]
    for idx, func in enumerate(funcs, 1):
        try:
            _, df_s, f = func()
            print(f"Case {idx} OK! Shape: {df_s.shape}, Features: {f}")
        except Exception as e:
            print(f"Case {idx} ERROR: {e}")
