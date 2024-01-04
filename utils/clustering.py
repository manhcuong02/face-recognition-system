import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Union
# Tạo dữ liệu ngẫu nhiên

def clustering_embedding_vectors(embedding_vectors: Union[np.ndarray, torch.Tensor], n_clusters = 1, n_init = 10) -> np.ndarray:
    if isinstance(embedding_vectors, torch.Tensor):
        embedding_vectors = embedding_vectors.detach().cpu().numpy()

    kmeans = KMeans(n_clusters = n_clusters, n_init = n_init)

    kmeans.fit(embedding_vectors)

    centroids = kmeans.cluster_centers_
    
    if n_clusters == 1:
        return centroids[0]
    return centroids