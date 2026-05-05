import json
import os

cases = [
    {
        "filename": "notebooks/Case1_Finansial.ipynb",
        "title": "# Studi Kasus 1: Kapasitas Finansial & Skala Bisnis",
        "func": "preprocess_finansial",
        "desc": "Notebook ini berfokus mengelompokkan UMKM berdasarkan kekuatan finansial dan operasional (Omset, Bantuan, KUR, Karyawan, Lama Usaha)."
    },
    {
        "filename": "notebooks/Case2_Demografi.ipynb",
        "title": "# Studi Kasus 2: Demografi Pemilik vs Ketahanan Usaha",
        "func": "preprocess_demografi",
        "desc": "Notebook ini berfokus menganalisis pola antara profil pemilik (Usia, Jenis Kelamin, Pendidikan) dengan usia UMKM."
    },
    {
        "filename": "notebooks/Case3_Digitalisasi.ipynb",
        "title": "# Studi Kasus 3: Tingkat Digitalisasi & Jangkauan Pemasaran",
        "func": "preprocess_digitalisasi",
        "desc": "Notebook ini membedah adopsi teknologi UMKM (Media Elektronik), komoditas ekspor, dan jangkauan pasar."
    },
    {
        "filename": "notebooks/Case4_AsetModal.ipynb",
        "title": "# Studi Kasus 4: Kemandirian Aset & Pemodalan",
        "func": "preprocess_aset_modal",
        "desc": "Notebook ini berfokus pada kemandirian UMKM melalui analisis kepemilikan aset dan status bantuan permodalan."
    },
    {
        "filename": "notebooks/Case5_Ketenagakerjaan.ipynb",
        "title": "# Studi Kasus 5: Profil Ketenagakerjaan",
        "func": "preprocess_ketenagakerjaan",
        "desc": "Notebook ini mengelompokkan UMKM berdasarkan kesejahteraan pekerja, rasio gender, usia pekerja, dan asuransi BPJS."
    }
]

notebook_template = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

for case in cases:
    nb = json.loads(json.dumps(notebook_template))
    
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            case["title"] + "\n\n",
            case["desc"] + "\n\n",
            "**Metode**: K-Means Clustering + Cuckoo Search Algorithm Optimization\n"
        ]
    })
    
    code = f"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("../src"))
from preprocessing import {case['func']}
from csa_core import hitung_optimal_k_elbow, cuckoo_search_kmeans, final_kmeans, plot_hasil_cluster, evaluasi_kualitas_klasterisasi

# 1. Preprocessing Data Khusus Skenario
df_raw, df_scaled, list_fitur = {case['func']}("../data/Data Set UMKM.xlsx")
X_scaled = df_scaled.values

print(f"Data shape: {{X_scaled.shape}}")
print(f"Fitur: {{list_fitur}}")
df_raw.head()
""".strip()
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.split("\n")]})
    
    code_elbow = """
# 2. Mencari K-Optimal secara Otomatis
print("\\nMencari jumlah klaster (K) optimal...")
optimal_k = hitung_optimal_k_elbow(X_scaled, max_k=8)
""".strip()
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code_elbow.split("\n")]})

    code_train = """
# 3. Proses Training (Fitting) dengan Cuckoo Search Algorithm
print(f"\\nMemulai Cuckoo Search dengan K={optimal_k}...")
best_nest, fitness_history = cuckoo_search_kmeans(
    X=X_scaled, 
    k=optimal_k, 
    n_nests=10, 
    max_iter=30, 
    pa=0.25
)

# 4. Fine-Tuning dengan K-Means konvensional
print("\\nMelakukan Fine-Tuning K-Means...")
labels, final_centroids = final_kmeans(X_scaled, best_nest)
""".strip()
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code_train.split("\n")]})

    code_eval = """
# 5. Evaluasi Metrik & Visualisasi
evaluasi_kualitas_klasterisasi(X_scaled, labels, final_centroids)

# Memasukkan hasil klaster ke dataframe asli
df_raw["Cluster"] = labels
print("\\nDistribusi Klaster:")
print(df_raw["Cluster"].value_counts())

# Plot hasil
plot_hasil_cluster(X_scaled, final_centroids, labels, list_fitur)
""".strip()
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code_eval.split("\n")]})

    with open(case["filename"], "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    
    print(f"Generated {case['filename']}")
