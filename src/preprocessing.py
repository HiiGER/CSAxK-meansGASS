import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def clean_and_preprocess_data(filepath='data/Data Set UMKM.xlsx'):
    """
    Memuat, membersihkan, dan melakukan ekstraksi fitur untuk dataset UMKM.
    Fokus utama: Kapasitas Finansial (Omset & Bantuan).
    """
    print("-> Memuat dataset...")
    # Header di baris 1 (index 1), karena baris 0 sepertinya aneh
    df = pd.read_excel(filepath, header=1)
    
    # 1. Pilih Kolom yang Relevan untuk Kapasitas Finansial & Operasional
    cols_to_keep = [
        'Tanggal Pendirian Usaha',
        'Modal Bantuan Pemerintah',
        'Pinjaman Kredit Usaha Rakyat',
        'Omset per-Tahun',
        'Laki-laki', # Karyawan
        'Perempuan', # Karyawan
    ]
    
    # Kita buat salinan dataframe dengan kolom terpilih
    df_clean = df[cols_to_keep].copy()
    
    # 2. Ekstraksi "Lama Usaha" dari "Tanggal Pendirian Usaha"
    print("-> Mengekstrak umur usaha...")
    def extract_year(date_str):
        if pd.isna(date_str) or str(date_str).strip() == '-' or str(date_str).strip() == '':
            return np.nan
        # Coba ambil 4 digit tahun (regex)
        import re
        match = re.search(r'\d{4}', str(date_str))
        if match:
            return int(match.group(0))
        return np.nan

    df_clean['Tahun Pendirian'] = df_clean['Tanggal Pendirian Usaha'].apply(extract_year)
    current_year = datetime.now().year
    df_clean['Lama Usaha (Tahun)'] = current_year - df_clean['Tahun Pendirian']
    
    # Isi NaN pada Lama Usaha dengan median
    median_lama_usaha = df_clean['Lama Usaha (Tahun)'].median()
    df_clean['Lama Usaha (Tahun)'] = df_clean['Lama Usaha (Tahun)'].fillna(median_lama_usaha)
    
    # 3. Transformasi Bantuan & Pinjaman (Biner: 1 jika ada tulisan, 0 jika '-')
    print("-> Mengubah data Bantuan & Pinjaman menjadi biner (0/1)...")
    def to_binary(val):
        if pd.isna(val) or str(val).strip() == '-' or str(val).strip() == '' or str(val).lower() == 'tidak':
            return 0
        return 1
        
    df_clean['Menerima Bantuan'] = df_clean['Modal Bantuan Pemerintah'].apply(to_binary)
    df_clean['Punya Pinjaman KUR'] = df_clean['Pinjaman Kredit Usaha Rakyat'].apply(to_binary)
    
    # 4. Encoding Omset per-Tahun (Ordinal)
    print("-> Melakukan encoding tingkatan Omset per-Tahun...")
    # Pastikan kategori unik dari data asli (Ini perkiraan asumsi, sesuaikan jika ada typo di data asli)
    omset_mapping = {
        'Kurang dari 10 juta': 1,
        '10 - 50 Juta': 2,
        '50 - 300 Juta': 3,
        'Lebih dari 300 Juta': 4
    }
    
    def map_omset(val):
        val_str = str(val).strip()
        for key, num in omset_mapping.items():
            if key.lower() in val_str.lower():
                return num
        return 1 # Default ke paling rendah jika tidak diketahui atau '-'
        
    df_clean['Tingkat Omset'] = df_clean['Omset per-Tahun'].apply(map_omset)
    
    # 5. Menghitung Total Karyawan
    print("-> Menghitung total karyawan...")
    # Ubah '-' jadi 0
    df_clean['Laki-laki'] = pd.to_numeric(df_clean['Laki-laki'].replace('-', 0), errors='coerce').fillna(0)
    df_clean['Perempuan'] = pd.to_numeric(df_clean['Perempuan'].replace('-', 0), errors='coerce').fillna(0)
    df_clean['Total Karyawan'] = df_clean['Laki-laki'] + df_clean['Perempuan']
    
    # === DATAFRAME AKHIR UNTUK CLUSTERING ===
    final_features = [
        'Lama Usaha (Tahun)',
        'Menerima Bantuan',
        'Punya Pinjaman KUR',
        'Tingkat Omset',
        'Total Karyawan'
    ]
    
    df_final = df_clean[final_features].copy()
    
    # 6. Scaling Data (Sangat Penting untuk K-Means)
    print("-> Melakukan Feature Scaling menggunakan StandardScaler...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_final)
    
    # Kembalikan ke dataframe untuk mempermudah melihat nama kolom
    df_scaled = pd.DataFrame(scaled_data, columns=final_features)
    
    print("✅ Preprocessing Selesai!")
    return df_final, df_scaled, final_features

if __name__ == "__main__":
    df_raw, df_scaled, features = clean_and_preprocess_data()
    print("\n[Preview Data Asli (Numerik)]:")
    print(df_raw.head())
    print("\n[Preview Data Scaled (Siap masuk Model)]:")
    print(df_scaled.head())
    print(f"\nTotal Data: {df_scaled.shape[0]} baris, {df_scaled.shape[1]} fitur.")
