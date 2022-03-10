from sklearn.model_selection import KFold
from aif360.datasets import StandardDataset
from test_classifiers import ClassifierTester


def test_classifier_on_folds(X_biased, X_fair, categorical_attributes, number_of_folds):
    kf = KFold(n_splits=number_of_folds)
    for train_index, test_index in kf.split(X_biased):
        biased_data_train, fair_data_train = X_biased.iloc[train_index], X_fair.iloc[train_index]
        biased_data_test, fair_data_test = X_biased.iloc[test_index], X_fair.iloc[test_index]

        biased_data_train = StandardDataset(biased_data_train, label_name="Pass", favorable_classes=[1],
                                                         protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                                         categorical_features=categorical_attributes)

        fair_data_train = StandardDataset(fair_data_train, label_name="Pass", favorable_classes=[1],
                                                       protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                                       categorical_features=categorical_attributes)

        biased_data_test = StandardDataset(biased_data_test, label_name="Pass", favorable_classes=[1],
                                            protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                            categorical_features=categorical_attributes)

        fair_data_test = StandardDataset(fair_data_test, label_name="Pass", favorable_classes=[1],
                                          protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                          categorical_features=categorical_attributes)

        classifier_tester = ClassifierTester(fair_data_train, fair_data_test, biased_data_train, biased_data_test)
        performances = classifier_tester.test_all_algorithms(test_on_fair_test_set=True)
        print(performances)