import argparse
import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, List

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def entropy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    dot_product = np.clip(dot_product, 1e-12, None)  # Avoid negative values and log(0)
    normalized_dot_product = dot_product / np.sum(dot_product)
    return -np.sum(normalized_dot_product * np.log2(normalized_dot_product))

def calculate_mean_metric(selected_embedding: np.ndarray, embeddings: List[np.ndarray], metric: str = 'cosine') -> float:
    """
    Calculate the mean metric (cosine similarity or entropy) of the selected embedding with other embeddings.
    
    Args:
        selected_embedding: The embedding to compare with others.
        embeddings: List of embeddings.
        metric: Metric to use ('cosine' or 'entropy').
    
    Returns:
        Mean metric value.
    """
    similarities = []
    for embedding in embeddings:
        if not np.array_equal(embedding, selected_embedding):
            if metric == 'cosine':
                similarity = cosine_similarity(selected_embedding, embedding)
            elif metric == 'entropy':
                similarity = entropy(selected_embedding, embedding)
            similarities.append(similarity)
    return np.mean(similarities)

def bias_measurement(D: List[np.ndarray], e: np.ndarray) -> float:
    """
    Calculate the bias measurement for an embedding.
    
    Args:
        D: List of all embeddings.
        e: Embedding to calculate the bias for.
    
    Returns:
        Bias value for the embedding.
    """
    chance_level = 1 / len(D)
    mean_metric = calculate_mean_metric(e, D, metric='cosine')
    bias_value = np.log(mean_metric / chance_level)
    return bias_value

def initialize_centroids_knexus(X: np.ndarray, K: int) -> np.ndarray:
    indices = np.random.choice(X.shape[0], K, replace=False)
    return X[indices]

def assign_clusters_knexus(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    clusters = []
    for x in X:
        distances = np.array([np.linalg.norm(x - c) for c in centroids])
        clusters.append(np.argmin(distances))
    return np.array(clusters)

def update_centroids_knexus(X: np.ndarray, clusters: np.ndarray, K: int) -> np.ndarray:
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        cluster_points = X[clusters == k]
        if len(cluster_points) > 0:
            centroids[k] = np.mean(cluster_points, axis=0)
    return centroids

def knexus(X: np.ndarray, K: int, max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform clustering using the K-NEXUS algorithm.
    """
    centroids = initialize_centroids_knexus(X, K)
    for _ in range(max_iters):
        clusters = assign_clusters_knexus(X, centroids)
        new_centroids = update_centroids_knexus(X, clusters, K)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def main(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to perform clustering and select representative classes.
    
    Args:
        args: Command line arguments.
          - embeddings: a dictionary of class average embeddings with labels as keys
          - sample_counts: a dictionary of sample counts per class with labels as keys
          - num_clusters: number of clusters to create with K-NEXUS
    
    Returns:
        DataFrame of sorted clustering results and DataFrame of representative classes.
    """
    embeddings = args.embeddings
    sample_counts = args.sample_counts
    num_clusters = args.num_clusters

    embedding_values = np.array(list(embeddings.values()))
    classes = list(embeddings.keys())

    # Calculate bias values for all classes
    bias_values = [bias_measurement(embedding_values, embedding) for embedding in embedding_values]

    # Perform K-NEXUS clustering
    centroids, clusters = knexus(embedding_values, num_clusters)

    # Create a DataFrame for clustering results
    cluster_results_df = pd.DataFrame({
        'Class': classes,
        'Bias Value': bias_values,
        'Cluster': clusters,
        'Sample Count': [sample_counts[cls] for cls in classes]
    })

    # Sort the DataFrame by cluster and sample count for better visualization
    cluster_results_df_sorted = cluster_results_df.sort_values(by=['Cluster', 'Sample Count'], ascending=[True, False]).reset_index(drop=True)

    # Select representative class for each cluster
    representative_classes = cluster_results_df_sorted.groupby('Cluster').first().reset_index()

    return cluster_results_df_sorted, representative_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform clustering on embeddings and select representative classes.")
    parser.add_argument('--path', type=str, required=True, help='Path to JSON file containing class average embeddings (E-bar) and sample counts.')
    parser.add_argument('--num_clusters', type=int, default=150, help='Number of clusters')

    args = parser.parse_args()
    
    # Load embeddings and sample counts from JSON file
    # The JSON file should have the following format:
    # {
    #     "class1": {
    #         "embeddings": [0.1, 0.2, 0.3, ...],
    #         "sample_count": 100
    #     },
    #     ...
    # }
    with open(args.path, 'r') as f:
        data = json.load(f)
    args.embeddings = {key: np.array(value['embeddings']) for key, value in data.items()}
    args.sample_counts = {key: value['sample_count'] for key, value in data.items()}

    cluster_results_df_sorted, representative_classes = main(args)
    cluster_results_df_sorted.to_csv('cluster_results.csv', index=False)
    representative_classes.to_csv('representative_classes.csv', index=False)
