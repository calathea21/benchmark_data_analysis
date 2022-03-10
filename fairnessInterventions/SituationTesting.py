import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import numpy as np

class Situation_Testing:
    def __init__(self, k, threshold):
        self.k = k
        self.threshold = threshold

    def fit(self, X):
        train_data = pd.DataFrame(X.features, columns=X.feature_names)
        train_data = train_data.drop(train_data[X.protected_attribute_names], axis=1)

        sensitive_attribute = X.protected_attributes.ravel()
        labels = X.labels.ravel()

        protected_indices = np.where(sensitive_attribute==0)[0]
        unprotected_indices = np.where(sensitive_attribute==1)[0]

        non_positive_labels = np.where(labels==0)[0]
        possibly_discriminated = set(protected_indices).intersection(set(non_positive_labels))
        print("Number of possibly discriminated: " + str(len(possibly_discriminated)))

        distance_matrix = self.make_distance_matrix(train_data)
        self.discriminated_indices = []

        for i in possibly_discriminated:
            protected_neighbors, unprotected_neighbors = self.find_k_nearest_neighbors(i, distance_matrix, protected_indices, unprotected_indices)
            difference_in_treatement = self.calc_difference(protected_neighbors, unprotected_neighbors, labels)
            if difference_in_treatement >= self.threshold:
                self.discriminated_indices.append(i)
        return


    def transform(self, X):
        print("Number of actually discriminated: " + str(len(self.discriminated_indices)) )
        transformed_data = X.copy(deepcopy=True)
        new_class_labels = X.labels.copy()
        np.put(new_class_labels, self.discriminated_indices, 1)
        transformed_data.labels = new_class_labels
        return transformed_data


    def find_k_nearest_neighbors(self, i, distance_matrix, protected_indices, unprotected_indices):
        distance_row = distance_matrix.iloc[i]

        protected_instances = distance_row.iloc[protected_indices]
        unprotected_instances = distance_row.iloc[unprotected_indices]

        protected_neighbours_idx = np.argpartition(protected_instances, self.k + 1)
        unprotected_neighbours_idx = np.argpartition(unprotected_instances, self.k)

        protected_neighbours = (protected_instances.iloc[protected_neighbours_idx[1:self.k + 1]])
        unprotected_neighbours = (unprotected_instances.iloc[unprotected_neighbours_idx[:self.k]])

        # return all nearest neighbours except of i itself
        return (protected_neighbours, unprotected_neighbours)

    def calc_difference(self, protected_neighbours, unprotected_neighbours, class_info):
        proportion_positive_protected = sum(class_info[protected_neighbours.index]) / len(protected_neighbours)
        proportion_positive_unprotected = sum(class_info[unprotected_neighbours.index]) / len(
            unprotected_neighbours)
        return (proportion_positive_unprotected - proportion_positive_protected)

    def make_distance_matrix(self, data):
        dists = pdist(data, self.distance)
        distance_matrix = pd.DataFrame(squareform(dists), columns=data.index, index=data.index)
        return distance_matrix

    def distance(self, x, y):
        sum_of_distances = 0

        for i in range(len(x)):
            sum_of_distances += abs(x[i]-y[i])

        return sum_of_distances