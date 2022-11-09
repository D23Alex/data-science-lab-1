from sklearn.datasets import make_blobs
import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def load_default_library_configs():
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 8)


def get_number_of_clusters_using_elbow_method(k_means_criteria, number_of_clusters_represented_by_zero_element):
    if len(k_means_criteria) < 2:
        return len(k_means_criteria)
    for i in range(len(k_means_criteria) - 1):
        current_k_means = k_means_criteria[i]
        next_k_means = k_means_criteria[i + 1]
        current_to_next_ratio = current_k_means / next_k_means
        if current_to_next_ratio < 1.2:
            return i + number_of_clusters_represented_by_zero_element
    return number_of_clusters_represented_by_zero_element


def clusterize_toy_dataset():
    # CONFIG
    DBSCAN_params = [(1, 5), (1.75, 2)]
    n_samples = 100
    random_state = 38
    centers = 5
    toy_dataset_name = "Toy data"
    toy_x_axis_name = "Toy data x axis name"
    toy_y_axis_name = "Toy data y axis name"
    lowest_amount_of_clusters_guess = 2
    highest_amount_of_clusters_guess = 10

    toy_dataset, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=centers)
    display_dataset(toy_dataset, toy_dataset_name, toy_x_axis_name, toy_y_axis_name, None)
    apply_k_means_to_dataset(toy_dataset, lowest_amount_of_clusters_guess, highest_amount_of_clusters_guess, toy_dataset_name, toy_x_axis_name, toy_y_axis_name)
    apply_DBSCAN_to_dataset(toy_dataset, DBSCAN_params, toy_dataset_name, toy_x_axis_name, toy_y_axis_name)


def clusterize_mall_customers_dataset():
    # CONFIG
    DBSCAN_params = [(8, 3), (9, 2), (9, 3)]
    mall_customers_dataset_name = "Mall customers"
    mall_customers_x_axis_name = "Annual Income (k$)"
    mall_customers_y_axis_name = "Spending Score (1-100)"
    lowest_amount_of_clusters_guess = 2
    highest_amount_of_clusters_guess = 10

    data = pd.read_csv("Mall_Customers.csv")
    mall_customers_dataset = data[['Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :].values
    display_dataset(mall_customers_dataset, mall_customers_dataset_name, mall_customers_x_axis_name,
                    mall_customers_y_axis_name, None)
    apply_k_means_to_dataset(mall_customers_dataset, lowest_amount_of_clusters_guess, highest_amount_of_clusters_guess, mall_customers_dataset_name,
                             mall_customers_x_axis_name, mall_customers_y_axis_name)
    apply_DBSCAN_to_dataset(mall_customers_dataset, DBSCAN_params, mall_customers_dataset_name,
                            mall_customers_x_axis_name, mall_customers_y_axis_name)


def display_dataset(dataset, title, x_axis_name, y_axis_name, clustering):
    fig, ax = plt.subplots()
    configure_plot(ax, title, x_axis_name, y_axis_name)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=clustering)
    plt.show()


def apply_k_means_to_dataset(dataset, lowest_amount_of_clusters_guess, highest_amount_of_clusters_guess, dataset_name, x_axis_name, y_axis_name):
    criteria = get_criteria(dataset, highest_amount_of_clusters_guess, lowest_amount_of_clusters_guess)
    print("[data] List of k-means scores for dataset " + dataset_name + " - will be used by the elbow method:")
    print(criteria.__str__() + "\n")
    display_k_means_score_by_amount_of_clusters(criteria, highest_amount_of_clusters_guess,
                                                lowest_amount_of_clusters_guess, dataset_name)
    n_clusters = get_number_of_clusters_using_elbow_method(criteria, 2)
    print("[!] " + "Number of clusters for '" + dataset_name + "' was set to " + n_clusters.__str__() + " using the elbow method")
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_model.fit(dataset)
    labels = kmeans_model.labels_
    display_dataset(dataset, "k-means in action: applied for " + n_clusters.__str__() + " clusters of dataset '" + dataset_name + "'", x_axis_name, y_axis_name, labels)


def display_k_means_score_by_amount_of_clusters(criteria, highest_amount_of_clusters_guess,
                                                lowest_amount_of_clusters_guess, dataset_name):
    fig, ax = plt.subplots()
    configure_plot(ax, dataset_name + ": k-means score by number of clusters", "number of clusters", "k-means score")
    plt.plot(range(lowest_amount_of_clusters_guess, highest_amount_of_clusters_guess), criteria)
    plt.show()


def get_criteria(dataset, highest_amount_of_clusters_guess, lowest_amount_of_clusters_guess):
    criteria = []
    for k in range(lowest_amount_of_clusters_guess, highest_amount_of_clusters_guess):
        kmeans_model = KMeans(n_clusters=k, random_state=3)
        kmeans_model.fit(dataset)
        criteria.append(kmeans_model.inertia_)
    return criteria


def apply_DBSCAN_to_dataset(dataset, DBSCAN_params, dataset_name, x_axis_name, y_axis_name):
    for param in DBSCAN_params:
        clustering = DBSCAN(eps=param[0], min_samples=param[1]).fit_predict(dataset)
        print("[data] DBSCAN Clustering - done. result:")
        print(clustering)
        print("\n")
        display_dataset(dataset, "DBSCAN in action: " + dataset_name + "(eps = " + param[0].__str__() + " min_samples = " + param[1].__str__() + " )", x_axis_name, y_axis_name, clustering)


def configure_plot(ax, title, x_axis_name, y_axis_name):
    ax.set_xlabel("x axis: " + x_axis_name)
    ax.set_ylabel("y axis: " + y_axis_name)
    ax.set_title(title)


def main():
    load_default_library_configs()
    clusterize_toy_dataset()
    clusterize_mall_customers_dataset()


if __name__ == "__main__":
    main()

