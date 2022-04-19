import numpy as np
from scipy.optimize import minimize
import scipy.stats
import pandas as pd
import math
from scipy.spatial.distance import pdist, squareform


class WeightedEuclideanDistanceLearner():
    def __init__(self, standard_dataset, l1_norm):
        self.standard_dataset = standard_dataset
        self.protected_info = np.ravel(self.standard_dataset.protected_attributes)
        self.labels = np.ravel(self.standard_dataset.labels)

        protected_attribute_name = self.standard_dataset.protected_attribute_names[0]
        column_index_of_protected_attribute = self.standard_dataset.feature_names.index(protected_attribute_name)
        self.data = self.standard_dataset.features
        self.data = np.delete(self.data, column_index_of_protected_attribute, axis=1)
        self.number_of_attributes = self.data.shape[1]
        self.l1_norm = l1_norm

    # def __init__(self, data, protected_info, labels, indices_info, l1_norm):
    #     self.data = data
    #     self.number_of_attributes = data.shape[1]
    #     self.protected_info = protected_info
    #     self.labels = labels
    #     self.l1_norm = l1_norm
    #     self.indices_info = indices_info
    #     self.interval_indices = self.indices_info['interval']
    #     self.ordinal_indices = self.indices_info['ordinal']
    #
    #     self.labels_protected_instances = self.labels[np.where(self.protected_info == 0)]
    #     self.labels_unprotected_instances = self.labels[np.where(self.protected_info == 1)]
    #
    #     self.data_protected_instances = self.data[np.where(self.protected_info == 0)]
    #     self.data_unprotected_instances = self.data[np.where(self.protected_info == 1)]
    #
    #     self.prot_same_label, self.prot_diff_label = self.get_squared_diff_vectors_for_instances_with_same_and_different_class_label(
    #         self.labels_protected_instances, self.data_protected_instances)
    #     self.unprot_same_label, self.unprot_diff_label = self.get_squared_diff_vectors_for_instances_with_same_and_different_class_label(
    #         self.labels_unprotected_instances, self.data_unprotected_instances)
    #

    def give_squared_abs_difference_vector_between_instances(self, x, y):
        difference_vector = []
        for index in range(0, len(x)):
            difference_vector.append(abs(x[index] - y[index])**2)
        return np.array(difference_vector)


    def get_squared_diff_vectors_for_instances_with_same_and_different_class_label(self, labels, data):
        same_label_matrix = np.empty((0, self.number_of_attributes), float)
        different_label_matrix = np.empty((0, self.number_of_attributes), float)

        for i in range(0, len(data)):
            for j in range(i + 1, len(data)):
                if labels[i] != labels[j]:
                    abs_difference_vector = self.give_squared_abs_difference_vector_between_instances(data[i], data[j])
                    different_label_matrix = np.append(different_label_matrix, [abs_difference_vector], axis=0)
                else:
                    abs_difference_vector = self.give_squared_abs_difference_vector_between_instances(data[i], data[j])
                    same_label_matrix = np.append(same_label_matrix, [abs_difference_vector], axis=0)
        return same_label_matrix, different_label_matrix


    def calc_weighted_distances(self, weights, difference_matrix):
        distances = np.matmul(difference_matrix, weights)
        square_root_distances = np.sqrt(distances)
        return square_root_distances


    def objective(self, weights):
        distances_between_protected_with_diff_label = self.calc_weighted_distances(weights, self.prot_diff_label)
        distances_between_protected_with_same_label = self.calc_weighted_distances(weights, self.prot_same_label)

        distances_between_unprotected_with_diff_label = self.calc_weighted_distances(weights, self.unprot_diff_label)
        distances_between_unprotected_with_same_label = self.calc_weighted_distances(weights, self.unprot_same_label)

        mean_distance_prot_with_diff_label = sum(distances_between_protected_with_diff_label) / len(
            distances_between_protected_with_diff_label)

        mean_distance_prot_with_same_label = sum(distances_between_protected_with_same_label) / len(
            distances_between_protected_with_same_label)

        mean_distance_unprot_with_diff_label = sum(distances_between_unprotected_with_diff_label) / len(
            distances_between_unprotected_with_diff_label)
        mean_distance_unprot_with_same_label = sum(distances_between_unprotected_with_same_label) / len(
            distances_between_unprotected_with_same_label)

        l1_norm_regularizer = self.l1_norm * sum(weights**2)

        objective_evaluation = ((mean_distance_prot_with_same_label + mean_distance_unprot_with_same_label) - (
                    mean_distance_prot_with_diff_label + mean_distance_unprot_with_diff_label) + l1_norm_regularizer)

        print(objective_evaluation)
        return objective_evaluation

    def make_euclidean_derivative_per_label_group(self, label_group, weights):
        derivative_vector = []
        euclidean_distances = self.calc_weighted_distances(weights, label_group)
        for i in range(len(weights)):
            sum_of_elements = 0
            for element in range(len(euclidean_distances)):
                euclidean_distance = euclidean_distances[element]
                if euclidean_distance != 0:
                    sum_of_elements += ((1 / (2 * euclidean_distance)) * (label_group[element][i]))
            derivative_vector.append(sum_of_elements)
        return derivative_vector

    def derivative(self, weights):
        derivative_prot_same = np.array(self.make_euclidean_derivative_per_label_group(self.prot_same_label, weights))
        derivative_prot_diff = np.array(self.make_euclidean_derivative_per_label_group(self.prot_diff_label, weights))
        derivative_unprot_same = np.array(self.make_euclidean_derivative_per_label_group(self.unprot_same_label, weights))
        derivative_unprot_diff = np.array(self.make_euclidean_derivative_per_label_group(self.unprot_diff_label, weights))

        derivative_prot_same = 1 / (len(self.prot_same_label)) * derivative_prot_same
        derivative_prot_diff = 1 / (len(self.prot_diff_label)) * derivative_prot_diff
        derivative_unprot_same = 1 / (len(self.unprot_same_label)) * derivative_unprot_same
        derivative_unprot_diff = 1 / (len(self.unprot_diff_label)) * derivative_unprot_diff

        sum_derivative_same = derivative_prot_same + derivative_unprot_same
        sum_derivative_diff = derivative_prot_diff + derivative_unprot_diff

        derivative = (sum_derivative_same - sum_derivative_diff)

        for i in range(len(weights)):
            derivative[i] += 2 * self.l1_norm * weights[i]

        return derivative

    def solve_objective(self):
        self.labels_protected_instances = self.labels[np.where(self.protected_info == 0)]
        self.labels_unprotected_instances = self.labels[np.where(self.protected_info == 1)]

        self.data_protected_instances = self.data[np.where(self.protected_info == 0)]
        self.data_unprotected_instances = self.data[np.where(self.protected_info == 1)]

        self.prot_same_label, self.prot_diff_label = self.get_squared_diff_vectors_for_instances_with_same_and_different_class_label(
            self.labels_protected_instances, self.data_protected_instances)

        self.unprot_same_label, self.unprot_diff_label = self.get_squared_diff_vectors_for_instances_with_same_and_different_class_label(
            self.labels_unprotected_instances, self.data_unprotected_instances)

        print("Init done")

        initial_weights = [0.1] * self.number_of_attributes

        b = (0.1, float('inf'))
        bds = [b] * self.number_of_attributes

        sol = minimize(self.objective, initial_weights, method='SLSQP', jac=self.derivative, bounds=bds)

        print(sol)
        return sol['x']


