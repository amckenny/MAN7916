#!/usr/bin/env python3
"""
cluster_analysis.py

This script reads all text files (*.txt) from the directory:
    /groups/course.man7916/corpora/aussie/About

It preprocesses the texts, vectorizes them using TF–IDF, and then performs a clustering analysis
(using scikit-learn’s KMeans). The results are visualized in several ways:
    - PCA projection scatter plot
    - t-SNE projection scatter plot
    - Hierarchical clustering dendrogram
    - Bar charts of top terms for each cluster

All plots are saved as image files in the user's output directory.
"""

import os
import glob
import string
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

# For text preprocessing (stopwords)
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def read_files(directory):
    """
    Reads all .txt files from a given directory.
    Returns:
        docs: list of document texts
        filenames: list of file names corresponding to each document
    """
    file_paths = glob.glob(os.path.join(directory, '*.txt'))
    docs = []
    filenames = []
    for fp in file_paths:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                docs.append(f.read())
            filenames.append(os.path.basename(fp))
        except Exception as e:
            print(f"Error reading {fp}: {e}")
    return docs, filenames


def preprocess(text):
    """
    Simple preprocessing: lowercasing, punctuation removal, and stopword filtering.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize by whitespace and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)


def vectorize_docs(docs):
    """
    Vectorizes the list of documents using TF–IDF.
    Returns:
        vectors: the TF–IDF sparse matrix
        vectorizer: the fitted TfidfVectorizer (useful for interpreting features)
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    return vectors, vectorizer


def perform_kmeans(vectors, n_clusters=5):
    """
    Performs KMeans clustering.
    Returns:
        labels: cluster labels for each document
        kmeans: the fitted KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(vectors)
    labels = kmeans.labels_
    return labels, kmeans


def plot_pca(vectors, labels, filenames, output_file='pca_clusters.png'):
    """
    Reduces dimensions with PCA and plots the documents in 2D.
    Saves the figure as an image file.
    """
    # Convert sparse matrix to dense if needed
    if hasattr(vectors, "toarray"):
        data = vectors.toarray()
    else:
        data = vectors
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Projection of Document Clusters')
    plt.legend(*scatter.legend_elements(), title="Clusters", loc="best")

    # Optionally annotate each point with its filename
    for i, fname in enumerate(filenames):
        plt.annotate(fname, (components[i, 0], components[i, 1]), fontsize=8, alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"PCA plot saved to {output_file}")


def plot_tsne(vectors, labels, filenames, output_file='tsne_clusters.png'):
    """
    Reduces dimensions with t-SNE and plots the documents in 2D.
    Saves the figure as an image file.
    """
    if hasattr(vectors, "toarray"):
        data = vectors.toarray()
    else:
        data = vectors
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_results = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Projection of Document Clusters')
    plt.legend(*scatter.legend_elements(), title="Clusters", loc="best")

    for i, fname in enumerate(filenames):
        plt.annotate(fname, (tsne_results[i, 0], tsne_results[i, 1]), fontsize=8, alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"t-SNE plot saved to {output_file}")


def plot_dendrogram(vectors, filenames, output_file='dendrogram.png'):
    """
    Plots a hierarchical clustering dendrogram of the documents.
    Saves the figure as an image file.
    """
    if hasattr(vectors, "toarray"):
        data = vectors.toarray()
    else:
        data = vectors
    linked = linkage(data, method='ward')

    plt.figure(figsize=(12, 8))
    dendrogram(linked, labels=filenames, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Document')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Dendrogram saved to {output_file}")


def plot_top_terms_per_cluster(vectorizer, kmeans, n_top_terms=10, output_prefix='cluster_top_terms'):
    """
    For each cluster, plots a bar chart of the top terms based on the KMeans cluster centroid.
    Each figure is saved as an image file.
    """
    terms = vectorizer.get_feature_names_out()
    centers = kmeans.cluster_centers_

    for i, center in enumerate(centers):
        # Get indices of top n terms in this cluster
        top_indices = center.argsort()[::-1][:n_top_terms]
        top_terms = [terms[idx] for idx in top_indices]
        top_scores = center[top_indices]

        plt.figure(figsize=(8, 6))
        plt.bar(range(n_top_terms), top_scores, tick_label=top_terms, color='skyblue')
        plt.title(f"Top Terms for Cluster {i}")
        plt.xlabel("Term")
        plt.ylabel("Average TF–IDF Weight")
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_file = f"{output_prefix}_cluster_{i}.png"
        plt.savefig(output_file)
        plt.close()
        print(f"Top terms plot for cluster {i} saved to {output_file}")


def main():
    # Define input/output directories (adjust if needed)
    input_directory = '/groups/course.man7916/corpora/aussie/About'
    output_directory = Path.home() / 'MAN7916' / 'output' / 'cluster_analysis'
    os.makedirs(output_directory, exist_ok=True)

    # Read files
    print("Reading files...")
    docs, filenames = read_files(input_directory)
    if not docs:
        print("No documents found. Exiting.")
        return

    # Preprocess the documents
    print("Preprocessing documents...")
    preprocessed_docs = [preprocess(doc) for doc in docs]

    # Vectorize documents
    print("Vectorizing documents with TF–IDF...")
    vectors, vectorizer = vectorize_docs(preprocessed_docs)

    # Perform KMeans clustering (adjust n_clusters as needed)
    n_clusters = 5
    print(f"Clustering documents into {n_clusters} clusters...")
    labels, kmeans = perform_kmeans(vectors, n_clusters=n_clusters)

    # Save a text file with cluster assignments
    filename = output_directory/"cluster_assignments.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for fname, label in zip(filenames, labels):
            f.write(f"{fname}: Cluster {label}\n")
    print(f"Cluster assignments saved to {filename}")

    # Visualization 1: PCA projection
    plot_pca(vectors, labels, filenames, output_file=str(output_directory/'pca_clusters.png'))

    # Visualization 2: t-SNE projection
    plot_tsne(vectors, labels, filenames, output_file=str(output_directory/'tsne_clusters.png'))

    # Visualization 3: Hierarchical clustering dendrogram
    plot_dendrogram(vectors, filenames, output_file=str(output_directory/'dendrogram.png'))

    # Visualization 4: Bar charts of top terms per cluster
    plot_top_terms_per_cluster(vectorizer, kmeans, n_top_terms=10, output_prefix=str(output_directory/'cluster_top_terms'))

    print("Cluster analysis complete.")


if __name__ == '__main__':
    main()
