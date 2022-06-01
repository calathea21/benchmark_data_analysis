'''
Class implementing the Situation Testing algorithm as proposed by Luong et al. This is a bias mitigation algorithm,
which works by detecting discrimination in the training data, such that it can be removed and a classifier can be trained
on a debiased version of it. Discirmination detection works, by going through all protected instances with a negative
decision label, and comparing their label to their nearest unprotected neighbours. If a high portion of the unprotected
neighbors received a positive label, the protected instance is marked as discriminated and its label will be changed to
positive.
'''

import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import numpy as np
from fairnessInterventions.WE_Learner import WeightedEuclideanDistanceLearner
from math import sqrt

class Situation_Testing:
    def __init__(self, k, threshold, learn_distance_function=False, lambda_l1_norm=0):
        self.k = k
        self.threshold = threshold
        self.learn_distance_function = learn_distance_function
        self.lambda_l1_norm = lambda_l1_norm

    def fit(self, X):
        train_data = pd.DataFrame(X.features, columns=X.feature_names)
        train_data = train_data.drop(train_data[X.protected_attribute_names], axis=1)

        if self.learn_distance_function:
            we_learner = WeightedEuclideanDistanceLearner(X, self.lambda_l1_norm)
            self.weights = we_learner.solve_objective()

        else:
            self.weights = np.ones(train_data.shape[1])

        sensitive_attribute = X.protected_attributes.ravel()
        labels = X.labels.ravel()

        protected_indices = np.where(sensitive_attribute==0)[0]
        unprotected_indices = np.where(sensitive_attribute==1)[0]

        non_positive_labels = np.where(labels==0)[0]
        possibly_discriminated = set(protected_indices).intersection(set(non_positive_labels))

        distance_matrix = self.make_distance_matrix(train_data)
        self.discriminated_indices = []
        self.discrimination_scores = []

        for i in possibly_discriminated:
            protected_neighbors, unprotected_neighbors = self.find_k_nearest_neighbors(i, distance_matrix, protected_indices, unprotected_indices)
            difference_in_treatement = self.calc_difference(protected_neighbors, unprotected_neighbors, labels)
            if difference_in_treatement >= self.threshold:
                self.discrimination_scores.append(difference_in_treatement)
                self.discriminated_indices.append(i)
        return


    def transform(self, X):
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


    def weighted_euclidean_distance(self, x, y):
        sum_of_distances = 0
        for i in range(len(x)):
            sum_of_distances += self.weights[i] * ((x[i]-y[i])**2)
        return sqrt(sum_of_distances)


    def distance(self, x, y):
        sum_of_distances = 0

        for i in range(len(x)):
            sum_of_distances += abs(x[i]-y[i])

        return sum_of_distances


    def get_discriminated_instances(self):
        return self.discriminated_indices

    def get_discrimination_scores_of_discriminated_instances(self):
        return self.discrimination_scores