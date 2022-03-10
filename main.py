from discrimination_analysis import discrimination_analysis_association_rules, discrimination_analysis_decision_tree, discrimination_analysis_subgroup_discovery, general_statistics_favoured_vs_discriminated
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset
from data_preprocessing import preprocess_data, train_test_toStandardDataset, add_columns_from_original_data, load_data_with_biased_and_unbiased_grades
from test_classifiers import ClassifierTester
from visualization_performances import scatter_plot_accuracy_vs_difference_in_positive_label
from kfold_testing import toStandardDataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_with_predictions = load_data_with_biased_and_unbiased_grades()
    #add_columns_from_original_data(["G1", "G2"], data_with_predictions)
    X_biased, X_fair, cat_attributes = preprocess_data(data_with_predictions, change_by_fail_pred=False, one_hot_encoding=False)
    #discrimination_analysis_subgroup_discovery(X_fair, X_biased)
    toStandardDataset(X_biased, X_fair, cat_attributes, number_of_folds=3)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
