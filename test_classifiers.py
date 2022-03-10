from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from fairnessInterventions.Massaging import Massaging
from fairnessInterventions.SituationTesting import Situation_Testing
import pandas as pd
import numpy as np
np.random.seed(5)


class ClassifierTester():
    def __init__(self, X_fair_train, X_fair_test, X_biased_train, X_biased_test):
        self.X_fair_train = X_fair_train
        self.X_fair_test = X_fair_test
        self.X_biased_train = X_biased_train
        self.X_biased_test = X_biased_test

    def evaluation_on_labels(self, test_label_predictions, test_on_fair_set):
        if test_on_fair_set:
            ground_truth_labels = self.X_fair_test.labels.ravel()
            ground_truth_dataset = self.X_fair_test
        else:
            ground_truth_labels = self.X_biased_test.labels.ravel()
            ground_truth_dataset = self.X_biased_test

        test_pred_dataset = self.X_fair_test.copy()
        test_pred_dataset.labels = test_label_predictions

        privileged_groups = [{'sex': 1}]  # girls
        unprivileged_groups = [{'sex': 0}]  # boys

        accuracy = accuracy_score(ground_truth_labels, test_label_predictions)
        print("Overall accuracy: " + str(accuracy))

        metric_org_test = BinaryLabelDatasetMetric(test_pred_dataset, unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)

        diff_in_pos_ratio = metric_org_test.mean_difference()
        print("Difference in ratio of positive decision labels: " + str(diff_in_pos_ratio))

        metric_org_vs_pred = ClassificationMetric(ground_truth_dataset, test_pred_dataset,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
        print("Difference in true positive rates: " + str(metric_org_vs_pred.equal_opportunity_difference()))
        print("Difference in false positive rates: " + str(metric_org_vs_pred.false_positive_rate_difference()))

        print("Accuracy on unprotected group: " + str(metric_org_vs_pred.accuracy(privileged=True)))
        print("Accuracy on protected group: " + str(metric_org_vs_pred.accuracy(privileged=False)))

        return accuracy, diff_in_pos_ratio


    def test_basic_unfair_classifiers(self, test_on_fair_test_set):
        print("\n\nTESTING UNFAIR CLASSIFIERS")
        features_X_train = self.X_biased_train.features
        labels_X_train = self.X_biased_train.labels.ravel()

        performance_unfair_data_dataframe = self.test_all_standard_classifiers(features_X_train, labels_X_train, test_on_fair_test_set)
        performance_unfair_data_dataframe["Intervention"] = "No Intervention"
        return performance_unfair_data_dataframe


    def test_benchmark_fair_classifier(self, test_on_fair_test_set):
        print("\n\nTESTING FAIR CLASSIFIERS")
        features_X_train = self.X_fair_train.features
        labels_X_train = self.X_fair_train.labels.ravel()

        performance_fair_data_dataframe = self.test_all_standard_classifiers(features_X_train, labels_X_train, test_on_fair_test_set)
        performance_fair_data_dataframe["Intervention"] = "Trained on fair data"
        return performance_fair_data_dataframe


    def test_massaging(self, test_on_fair_test_set):
        masseuse = Massaging()
        masseuse.fit(self.X_biased_train)
        preprocessed_X_train = masseuse.transform(self.X_biased_train)

        print("\n\nTESTING MASSAGING")
        features_X_train = preprocessed_X_train.features
        labels_X_train = preprocessed_X_train.labels.ravel()

        performance_massaging_dataframe = self.test_all_standard_classifiers(features_X_train, labels_X_train, test_on_fair_test_set)
        performance_massaging_dataframe["Intervention"] = "Massaging"
        return performance_massaging_dataframe


    def test_situation_testing(self, test_on_fair_test_set):
        st = Situation_Testing(k=10, threshold=0.3)
        st.fit(self.X_biased_train)

        preprocessed_X_train = st.transform(self.X_biased_train)

        print("\n\nTESTING SITUATION TESTING")
        features_X_train = preprocessed_X_train.features
        labels_X_train = preprocessed_X_train.labels.ravel()

        performance_situation_testing_dataframe = self.test_all_standard_classifiers(features_X_train, labels_X_train, test_on_fair_test_set)
        performance_situation_testing_dataframe["Intervention"] = "Situation Testing"
        return performance_situation_testing_dataframe

    def test_all_standard_classifiers(self, X_train, y_train, test_on_fair_test_set):
        features_X_test = self.X_biased_test.features

        # logistic regression
        print("*** Testing LogisticRegression Classifier ***")
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)
        test_labels_pred_lr = lr.predict(features_X_test)
        lr_acc, lr_diff_in_pos = self.evaluation_on_labels(test_labels_pred_lr, test_on_fair_test_set)

        # random forest
        print("*** Testing RandomForest Classifier ***")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        test_labels_pred_rf = rf.predict(features_X_test)
        rf_acc, rf_diff_in_pos = self.evaluation_on_labels(test_labels_pred_rf, test_on_fair_test_set)

        # support vector machine
        print("*** Testing Support Vector Machine ***")
        svm = SVC()
        svm.fit(X_train, y_train)
        test_labels_pred_svm = svm.predict(features_X_test)
        svm_acc, svm_diff_in_pos = self.evaluation_on_labels(test_labels_pred_svm, test_on_fair_test_set)

        performances_of_interest = [{"Accuracy": lr_acc, "Discrimination Score": lr_diff_in_pos}, {"Accuracy": rf_acc, "Discrimination Score": rf_diff_in_pos}, {"Accuracy": svm_acc, "Discrimination Score": svm_diff_in_pos}]
        performance_dataframe = pd.DataFrame(performances_of_interest, index=["LR", "RF", "SVM"])
        return performance_dataframe


    def test_all_algorithms(self, test_on_fair_test_set=True):
        performance_upper_benchmark = self.test_benchmark_fair_classifier(test_on_fair_test_set)
        performance_lower_benchmark = self.test_basic_unfair_classifiers(test_on_fair_test_set)
        performance_massaging = self.test_massaging(test_on_fair_test_set)
        performance_situation_testing = self.test_situation_testing(test_on_fair_test_set)

        all_performances_dataframe = pd.concat([performance_upper_benchmark, performance_lower_benchmark, performance_massaging, performance_situation_testing])
        print(all_performances_dataframe)
        return all_performances_dataframe