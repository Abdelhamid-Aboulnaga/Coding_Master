import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

"""Here, I wanted to cluster multiple conformational microstates of a protein to two macrostates using the corresponding helical content and radius of gyration"""

#data_2d = a dataframe containing radius of gyration and helical content values for each conformation

def clustering_2d_data(data2d):

    # Applying k-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data2d)

    # Assigning cluster labels to each data point
    clusters = kmeans.labels_

    # Calculating the relative probability (frequency) of each macrostate
    unique, counts = np.unique(clusters, return_counts=True)
    macrostate_probability = dict(zip(unique, counts / len(clusters)))

    print(macrostate_probability)

    # Plotting 1D histograms for Radius of Gyration and Helical Content with two different colors representing the macrostates

    plt.figure(figsize=(12, 6))

    # 1D Histogram for Radius of Gyration
    plt.subplot(2, 2, 1)
    for cluster in np.unique(clusters):
        plt.hist(data2d[clusters == cluster, 1], bins=30, alpha=0.6, label=f'Macrostate {cluster}')
    plt.xlabel('Radius of Gyration')
    plt.ylabel('Frequency')
    plt.title('Radius of Gyration by Macrostate')
    plt.legend()

    # 1D Histogram for Helical Content
    plt.subplot(2, 2, 2)
    for cluster in np.unique(clusters):
        plt.hist(data2d[clusters == cluster, 0], bins=30, alpha=0.6, label=f'Macrostate {cluster}')
    plt.xlabel('Helical Content')
    plt.ylabel('Frequency')
    plt.title('Helical Content by Macrostate')
    plt.legend()

    plt.subplot(2, 2, 3)
    for cluster in np.unique(clusters):
        plt.hist2d(data2d[clusters == cluster, 0], data2d[clusters == cluster, 1], bins=30, alpha=0.6, label=f'Macrostate {cluster}')

    plt.xlabel('Helical Content')
    plt.ylabel('Radius of Gyration')
    plt.colorbar(label='Frequency')

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    data_2d=sys.argv[1]
    clustering_2d_data(data2d)
