from discrimination_analysis import discrimination_analysis_association_rules, discrimination_analysis_decision_tree, discrimination_analysis_subgroup_discovery, general_statistics_favoured_vs_discriminated
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset
from data_preprocessing import preprocess_data, load_data_with_biased_and_unbiased_grades, add_columns_from_original_data, change_by_grade_prediction, change_by_ranking_position
from test_classifiers import ClassifierTester
from visualization_performances import scatter_plot_accuracy_vs_difference_in_positive_label
from kfold_testing import test_classifier_on_folds, test_classifier_on_train_test_split
from fairnessInterventions.WE_Learner import WeightedEuclideanDistanceLearner

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, categorical_attributes = load_data_with_biased_and_unbiased_grades()
    #add_columns_from_original_data(["subject"], data_with_predictions)
    X_fair, X_biased = preprocess_data(data, False, categorical_attributes, change_by_ranking_position, threshold_rank_fail=7, threshold_rank_pass=2)
    #
    # print(X_fair['sex'])
    # fair_data_train = StandardDataset(X_fair, label_name="Pass", favorable_classes=[1],
    #                                     protected_attribute_names=['sex'], privileged_classes=[["F"]],
    #                                     categorical_features=categorical_attributes)
    # privileged_groups = [{'sex': 1}]  # girls
    # unprivileged_groups = [{'sex': 0}]  # boys
    # metric = BinaryLabelDatasetMetric(fair_data_train, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    # print(metric.base_rate(privileged=True))
    # print(metric.base_rate(privileged=False))
    # print(metric.base_rate())
    # print(metric.difference())
    # print(biased_data_train.feature_names)
    # distance_learner = WeightedEuclideanDistanceLearner(biased_data_train, 0.01)
    # distance_learner.solve_objective()

    test_classifier_on_folds(X_biased, X_fair, categorical_attributes, number_of_folds=10, fairness_measure="TPR_diff", performance_measure="F1")
    #test_classifier_on_train_test_split(X_biased, X_fair, cat_attributes)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
