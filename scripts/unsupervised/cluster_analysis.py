#!/usr/bin/env python3
"""Cluster the same documents using TF-IDF and OpenAI embeddings.
Only for use on Newton with the Aussie corpus. Requires an OpenAI API key in ~/MAN7916/.env.
"""

import csv
import os
import string
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk

nltk.download("stopwords", quiet=True)

import numpy as np
from dotenv import load_dotenv
from nltk.corpus import stopwords
from openai import OpenAI, OpenAIError
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

PROJECT_DIRECTORY = Path.home() / "MAN7916"
INPUT_DIRECTORY = PROJECT_DIRECTORY / "corpora" / "aussie" / "About"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output" / "cluster_analysis"
N_CLUSTERS = 5

OPENAI_MODEL = "text-embedding-3-small"
SEED = 42
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def read_files(directory):
    docs, filenames = [], []
    for path in sorted(directory.glob("*.txt")):
        doc = path.read_text(encoding="utf-8")
        docs.append(doc)
        filenames.append(path.name)
    return docs, filenames


def vectorize_tfidf(docs):
    stop_words = set(stopwords.words("english"))
    processed_docs = []
    for doc in docs:
        tokens = doc.lower().translate(PUNCTUATION_TABLE).split()
        processed_docs.append(
            " ".join(word for word in tokens if word not in stop_words)
        )

    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(processed_docs), vectorizer.get_feature_names_out()


def embed_openai(docs, filenames):
    env_file = PROJECT_DIRECTORY / ".env"
    load_dotenv(env_file)
    api_key = os.getenv("OPENAI_KEY", "").strip()
    if not api_key:
        raise ValueError(f'Set OPENAI_KEY="..." in {env_file}.')

    vectors = []
    with OpenAI(api_key=api_key) as client:
        for doc, filename in zip(docs, filenames):
            response = client.embeddings.create(
                model=OPENAI_MODEL, input=doc, encoding_format="float"
            )
            vectors.append(response.data[0].embedding)
    return np.asarray(vectors)


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def plot_projection(coordinates, labels, filenames, method, projection, output_file):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab20").colors
    for label in np.unique(labels):
        color = colors[label % len(colors)]
        points = coordinates[labels == label]
        ax.scatter(
            points[:, 0], points[:, 1], label=f"Cluster {label}", color=color, s=50
        )
    for filename, (x, y) in zip(filenames, coordinates):
        ax.annotate(filename, (x, y), fontsize=8, alpha=0.7)
    ax.set(
        xlabel=f"{projection} component 1",
        ylabel=f"{projection} component 2",
        title=f"{method} | {projection} projection",
    )
    ax.legend(title="K-means")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def plot_dendrogram(data, filenames, method, output_file):
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linkage(data, method="ward"), labels=filenames, leaf_rotation=90, ax=ax)
    ax.set(title=f"{method} | Ward hierarchical clustering", ylabel="Distance")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def plot_top_terms(tfidf_vectors, terms, labels, method, output_directory):
    for plot in output_directory.glob("cluster_top_terms_cluster_*.png"):
        plot.unlink()

    for label in np.unique(labels):
        weights = np.asarray(tfidf_vectors[labels == label].mean(axis=0)).ravel()
        top = weights.argsort()[::-1][:10]
        top = top[weights[top] > 0]
        if not len(top):
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(terms[top][::-1], weights[top][::-1], color="skyblue")
        ax.set(
            title=f"{method} | Top terms for cluster {label}",
            xlabel="Mean TF-IDF weight",
        )
        fig.tight_layout()
        fig.savefig(output_directory / f"cluster_top_terms_cluster_{label}.png")
        plt.close(fig)


def cluster_documents(vectors, filenames, n_clusters, method, output_directory):
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED).fit_predict(
        vectors
    )
    output_directory.mkdir(exist_ok=True)
    write_csv(
        output_directory / "cluster_assignments.csv",
        ["filename", "cluster"],
        zip(filenames, labels),
    )

    data = vectors.toarray() if issparse(vectors) else vectors
    if data.shape[1] == 1:
        data = np.pad(data, ((0, 0), (0, 1)))

    pca = PCA(n_components=2, random_state=SEED).fit_transform(data)
    tsne = TSNE(
        n_components=2, perplexity=min(5, len(filenames) - 1), random_state=SEED
    )
    plot_projection(
        pca, labels, filenames, method, "PCA", output_directory / "pca_clusters.png"
    )
    plot_projection(
        tsne.fit_transform(data),
        labels,
        filenames,
        method,
        "t-SNE",
        output_directory / "tsne_clusters.png",
    )
    plot_dendrogram(data, filenames, method, output_directory / "dendrogram.png")
    return labels


def compare_clusters(filenames, tfidf_labels, openai_labels, output_directory):
    output_directory.mkdir(exist_ok=True)
    tfidf_ids = np.unique(tfidf_labels)
    openai_ids = np.unique(openai_labels)
    overlap = contingency_matrix(tfidf_labels, openai_labels)

    rows, columns = linear_sum_assignment(overlap, maximize=True)
    openai_to_tfidf = {
        openai_ids[col]: tfidf_ids[row]
        for row, col in zip(rows, columns)
        if overlap[row, col] > 0
    }
    aligned = [openai_to_tfidf.get(label) for label in openai_labels]
    matches = [bool(tfidf == matched) for tfidf, matched in zip(tfidf_labels, aligned)]
    write_csv(
        output_directory / "document_comparison.csv",
        [
            "filename",
            "tfidf_cluster",
            "openai_cluster",
            "openai_cluster_aligned_to_tfidf",
            "same_after_alignment",
        ],
        zip(filenames, tfidf_labels, openai_labels, aligned, matches),
    )
    write_csv(
        output_directory / "cluster_overlap.csv",
        ["tfidf_cluster", *(f"openai_cluster_{label}" for label in openai_ids)],
        ((label, *counts) for label, counts in zip(tfidf_ids, overlap)),
    )

    ari = adjusted_rand_score(tfidf_labels, openai_labels)
    matched = sum(matches)
    summary = (
        f"Documents compared: {len(filenames)}\n"
        f"Clusters found: {len(tfidf_ids)} TF-IDF, {len(openai_ids)} OpenAI\n"
        f"Adjusted Rand index: {ari:.3f}\n"
        f"Documents in matched clusters: {matched}/{len(filenames)} ({matched / len(filenames):.1%})\n"
        "\nThe adjusted Rand index compares document groupings regardless of cluster numbers.\n"
        "A score of 1 means identical groupings. Scores near 0 indicate chance-level agreement.\n"
    )
    (output_directory / "agreement_summary.txt").write_text(summary, encoding="utf-8")
    print(summary.split("\n\n", 1)[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(overlap, cmap="Blues", aspect="auto")
    midpoint = (overlap.min() + overlap.max()) / 2
    for (row, col), count in np.ndenumerate(overlap):
        color = "white" if count > midpoint else "black"
        ax.text(col, row, str(count), ha="center", va="center", color=color)
    ax.set_xticks(range(len(openai_ids)), openai_ids)
    ax.set_yticks(range(len(tfidf_ids)), tfidf_ids)
    ax.set(
        xlabel="OpenAI cluster",
        ylabel="TF-IDF cluster",
        title="Documents shared by each cluster pair",
    )
    fig.colorbar(heatmap, ax=ax, label="Documents")
    fig.tight_layout()
    fig.savefig(output_directory / "cluster_overlap.png")
    plt.close(fig)


def main():
    docs, filenames = read_files(INPUT_DIRECTORY)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {OUTPUT_DIRECTORY}", flush=True)

    # 1. Cluster the TF-IDF vectors.
    tfidf_vectors, terms = vectorize_tfidf(docs)
    tfidf_directory = OUTPUT_DIRECTORY / "tfidf"
    tfidf_labels = cluster_documents(
        tfidf_vectors, filenames, N_CLUSTERS, "TF-IDF", tfidf_directory
    )
    plot_top_terms(tfidf_vectors, terms, tfidf_labels, "TF-IDF", tfidf_directory)

    # 2. Cluster the OpenAI embeddings.
    print(f"Embedding {len(docs)} documents with {OPENAI_MODEL}...", flush=True)
    openai_vectors = embed_openai(docs, filenames)
    openai_directory = OUTPUT_DIRECTORY / "openai"
    openai_labels = cluster_documents(
        openai_vectors, filenames, N_CLUSTERS, "OpenAI", openai_directory
    )
    plot_top_terms(tfidf_vectors, terms, openai_labels, "OpenAI", openai_directory)

    # 3. Compare the two sets of cluster assignments.
    compare_clusters(
        filenames, tfidf_labels, openai_labels, OUTPUT_DIRECTORY / "comparison"
    )


if __name__ == "__main__":
    main()
