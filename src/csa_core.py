import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===============================
# FUNGSI DASAR Jarak & Fitness
# ===============================

def calculate_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def evaluate_sse(data, centroids):
    sse = 0
    for i in range(len(data)):
        distances = [calculate_distance(data[i], c) for c in centroids]
        min_dist = np.min(distances)
        sse += (min_dist ** 2)
    return sse

def evaluate_fitness(sse):
    return 1.0 / (1.0 + sse)

def levy_flight(Lambda=1.5):
    sigma_u = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) / \
               (math.gamma((1 + Lambda) / 2) * Lambda * (2 ** ((Lambda - 1) / 2)))) ** (1 / Lambda)
    sigma_v = 1
    
    u = np.random.normal(0, sigma_u, 1)
    v = np.random.normal(0, sigma_v, 1)
    
    step = u / (abs(v) ** (1 / Lambda))
    return step[0]

# ===============================
# K-OPTIMAL ELBOW (K-Means)
# ===============================

def hitung_optimal_k_elbow(data, max_k=10):
    sse = []
    K = range(1, max_k + 1)
    
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(data)
        sse.append(km.inertia_)  

    # Mencari elbow secara otomatis menggunakan kalkulasi titik terjauh dari garis start-end (segitiga)
    p1 = np.array([K[0], sse[0]])
    p2 = np.array([K[-1], sse[-1]])
    
    distances = []
    for i in range(len(K)):
        p_point = np.array([K[i], sse[i]])
        dist = np.linalg.norm(np.cross(p2 - p1, p1 - p_point)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
        
    best_k = K[np.argmax(distances)]
    
    plt.figure(figsize=(8, 4))
    plt.plot(K, sse, 'bx-')
    plt.vlines(best_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red', label=f'Optimal K = {best_k}')
    plt.xlabel('Jumlah Klaster (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Metode L-Bow Terotomatisasi (Pencarian Nilai K Optimal)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_k

# ===============================
# CUCKOO SEARCH ALGORITHM
# ===============================

def cuckoo_search_kmeans(X, k=3, n_nests=5, max_iter=20, pa=0.25):
    n_features = X.shape[1] 
    n_samples = X.shape[0]
    
    min_bounds = np.min(X, axis=0)
    max_bounds = np.max(X, axis=0)
    
    nests = []
    for _ in range(n_nests):
        nest_centroids = np.random.uniform(min_bounds, max_bounds, (k, n_features))
        nests.append(nest_centroids)
    nests = np.array(nests)
    
    fitness_history = []
    best_nest = None
    best_fitness = -1
    
    for iteration in range(max_iter):
        fitness_values = []
        
        for i in range(n_nests):
            sse = evaluate_sse(X, nests[i])
            fitness = evaluate_fitness(sse)
            fitness_values.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_nest = np.copy(nests[i])
                
        fitness_values = np.array(fitness_values)
        fitness_history.append((iteration+1, 1.0/best_fitness - 1.0, best_fitness))
        
        for i in range(n_nests):
            step_size = levy_flight() * 0.01
            new_nest = nests[i] + step_size * (nests[i] - best_nest)
            
            new_sse = evaluate_sse(X, new_nest)
            new_fitness = evaluate_fitness(new_sse)
            
            if new_fitness > fitness_values[i]:
                nests[i] = new_nest
                fitness_values[i] = new_fitness
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_nest = np.copy(new_nest)
                    
        worst_indices = np.argsort(fitness_values)
        num_abandon = int(pa * n_nests)
        
        for i in range(num_abandon):
            idx = worst_indices[i]
            rand_nest_idx = np.random.randint(0, n_nests)
            rand_step = np.random.uniform(0, 1) 
            nests[idx] = nests[idx] + rand_step * (best_nest - nests[rand_nest_idx])
            
    return best_nest, fitness_history

def final_kmeans(X, initial_centroids):
    centroids = np.copy(initial_centroids)
    clusters = np.zeros(len(X))
    
    while True:
        new_clusters = np.zeros(len(X))
        for i in range(len(X)):
            distances = [calculate_distance(X[i], c) for c in centroids]
            new_clusters[i] = np.argmin(distances)
            
        if np.array_equal(clusters, new_clusters):
            break 
            
        clusters = new_clusters
        for j in range(len(centroids)):
            cluster_points = X[clusters == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)
                
    return clusters, centroids


# ===============================
# VISUALISASI DINAMIS (2D dan 3D)
# ===============================

def plot_hasil_cluster(X_plot, centroids, labels, list_fitur):
    \"\"\"\n    Membuat grafik plotter untuk dataset secara otomatis mendeteksi dimensi: \n    Akan menggunakan 3D murni apabila dataset >= 3D.\n    \"\"\"
    n_dim = len(list_fitur)
    num_clusters = len(np.unique(labels))
    cmap = cm.get_cmap('Set1', num_clusters)
    
    fig = plt.figure(figsize=(10, 8))
    
    if n_dim == 2:
        ax = fig.add_subplot(111)
        for idx, cluster_label in enumerate(sorted(np.unique(labels))):
            cluster_idx = labels == cluster_label
            ax.scatter(X_plot[cluster_idx, 0], X_plot[cluster_idx, 1], color=cmap(idx), label=f'Cluster {cluster_label}', s=100)
        
        ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=250, label='Centroids Akhir')
        ax.set_xlabel(list_fitur[0] + ' (Scaled)')
        ax.set_ylabel(list_fitur[1] + ' (Scaled)')
        ax.set_title('Pemetaan Clustering (Pandangan 2D Klasik)')
        
    elif n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        for idx, cluster_label in enumerate(sorted(np.unique(labels))):
            cluster_idx = labels == cluster_label
            ax.scatter(X_plot[cluster_idx, 0], X_plot[cluster_idx, 1], X_plot[cluster_idx, 2], color=cmap(idx), label=f'Cluster {cluster_label}', s=100)
        
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='X', s=350, label='Centroids Akhir')
        ax.set_xlabel(list_fitur[0] + ' (Scaled)')
        ax.set_ylabel(list_fitur[1] + ' (Scaled)')
        ax.set_zlabel(list_fitur[2] + ' (Scaled)')
        ax.set_title('Pemetaan Clustering Aktual 3D (X, Y, Z Murni)')
        ax.view_init(elev=20., azim=30)
        
    elif n_dim >= 4:
        # Gunakan PCA 3D untuk memampatkan ke dalam persepsi Spasial 3-Sumbu agar tak terlewat
        print(f\"-> Dimensi {n_dim}D terdeteksi! Mengecilkan ke Ruang Visual 3D dengan Principal Component Analysis (PCA).\")
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_plot)
        centroids_pca = pca.transform(centroids)
        
        ax = fig.add_subplot(111, projection='3d')
        for idx, cluster_label in enumerate(sorted(np.unique(labels))):
            cluster_idx = labels == cluster_label
            ax.scatter(X_pca[cluster_idx, 0], X_pca[cluster_idx, 1], X_pca[cluster_idx, 2], color=cmap(idx), label=f'Cluster {cluster_label}', s=100)
        
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], c='black', marker='X', s=350, label='Centroids Akhir (PCA 3D)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f'Pemetaan Clustering Spasial (Reduksi PCA {n_dim}D -> 3D)')
        ax.view_init(elev=25., azim=45)

    # General configuration
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
