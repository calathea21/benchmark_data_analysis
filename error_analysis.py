import pandas as pd
import pysubgroup as ps
from fairnessInterventions.Massaging import Massaging
from fairnessInterventions.SituationTesting import Situation_Testing
from sklearn.model_selection import KFold, train_test_split
from aif360.datasets import StandardDataset
import pandas as pd
import numpy as np
np.random.seed(4)

class ErrorAnalyzer():
    def __init__(self, X_fair_train, X_fair_test, X_biased_train, X_biased_test):
        self.X_fair_train = X_fair_train
        self.X_fair_test = X_fair_test
        self.X_biased_train = X_biased_train
        self.X_biased_test = X_biased_test

    def favoured_discrimination_condition(self, data):
        if (data["FairLabel"] == True) and (data["Pass"] == False):
            return "Discriminated"
        elif (data["FairLabel"] == False) and (data["Pass"] == True):
            return "Favoured"
        elif (data["FairLabel"] == False) and (data["Pass"] == False):
            return "No Change in Negative Label"
        else:
            return "No Change in Positive Label"

    def add_discrimination_favoured_label(self, fair_data, biased_data):
        biased_data["FairLabel"] = fair_data["Pass"]
        biased_data["Discrimination_Label"] = biased_data.apply(self.favoured_discrimination_condition, axis=1)

        return biased_data

    def get_discriminated_and_favoured_instances(self):
        biased_labels = self.X_biased_train.labels
        fair_labels = self.X_fair_train.labels

        biased_labels_dataframe = pd.DataFrame(biased_labels)
        fair_labels_dataframe = pd.DataFrame(fair_labels)

        biased_labels_dataframe.columns =['Pass']
        fair_labels_dataframe.columns = ['Pass']

        biased_discrimination_labels = self.add_discrimination_favoured_label(fair_labels_dataframe, biased_labels_dataframe)
        return biased_discrimination_labels

    def test_discrimination_detection_of_intervention(self, intervention):
        if (intervention == "Massaging"):
            model = self.test_massaging()
            discriminated_instances_according_to_intervention = model.get_promotion_candidates()
        elif (intervention == "Situation Testing"):
            model = self.test_situation_testing()
            discriminated_instances_according_to_intervention = model.get_discriminated_instances()

        discrimination_labels_actual_data = self.get_discriminated_and_favoured_instances()
        discriminated_instances_actual_data = discrimination_labels_actual_data[
            discrimination_labels_actual_data["Discrimination_Label"] == "Discriminated"].index
        no_change_in_neg_instances_actual_data = discrimination_labels_actual_data[
            discrimination_labels_actual_data["Discrimination_Label"] == "No Change in Negative Label"].index

        correctly_detected_discriminated_indices = set(discriminated_instances_according_to_intervention).intersection(
            set(discriminated_instances_actual_data))
        incorrectly_detected_discriminated_indices = set(discriminated_instances_according_to_intervention).intersection(
            set(no_change_in_neg_instances_actual_data))
        missed_discriminated_indices = set(discriminated_instances_actual_data).difference(set(discriminated_instances_according_to_intervention))

        print("Number of correctly detected discriminated instances: " + str(len(correctly_detected_discriminated_indices)))
        print("Number of incorrectly detected discriminated instances: " + str(len(incorrectly_detected_discriminated_indices)))
        print("Number of missed discriminated instances: " + str(len(missed_discriminated_indices)))

        train_data_as_panda = pd.DataFrame(self.X_biased_train.features, columns=self.X_biased_train.feature_names)
        train_data_as_panda["romantic"] = train_data_as_panda[['romantic=no', 'romantic=yes']].idxmax(axis=1)
        train_data_as_panda["reason"] = train_data_as_panda[['reason=home', 'reason=other', 'reason=reputation', 'reason=course']].idxmax(axis=1)
        train_data_as_panda = train_data_as_panda.drop(['reason=home', 'reason=other', 'reason=reputation', 'reason=course', 'romantic=no', 'romantic=yes'], axis=1)
        print(train_data_as_panda)


        correctly_detected_discriminated_instances = train_data_as_panda.loc[correctly_detected_discriminated_indices]
        correctly_detected_discriminated_instances["Error Label"] = "Correctly Discriminated"
        incorrectly_detected_discriminated_instances = train_data_as_panda.loc[
            incorrectly_detected_discriminated_indices]
        incorrectly_detected_discriminated_instances["Error Label"] = "Incorrectly Discriminated"
        missed_discriminated_instances = train_data_as_panda.loc[missed_discriminated_indices]
        missed_discriminated_instances["Error Label"] = "Missed Discriminated"
        discrimination_data_and_errors = pd.concat([correctly_detected_discriminated_instances, incorrectly_detected_discriminated_instances, missed_discriminated_instances], ignore_index=True, axis=0)
        self.error_analysis_subgroup_discovery(discrimination_data_and_errors, "Missed Discriminated")
        self.error_analysis_subgroup_discovery(discrimination_data_and_errors, "Correctly Discriminated")
        self.error_analysis_subgroup_discovery(discrimination_data_and_errors, "Incorrectly Discriminated")



    def test_favouritism_detection_of_intervention(self, intervention="Massaging"):
        model = self.test_massaging()
        favoured_instances_according_to_intervention = model.get_demotion_candidates()

        discrimination_labels_actual_data = self.get_discriminated_and_favoured_instances()
        favoured_instances_actual_data = discrimination_labels_actual_data[
            discrimination_labels_actual_data["Discrimination_Label"] == "Favoured"].index
        no_change_in_pos_instances_actual_data = discrimination_labels_actual_data[
            discrimination_labels_actual_data["Discrimination_Label"] == "No Change in Positive Label"].index

        correctly_detected_favoured_indices = set(favoured_instances_according_to_intervention).intersection(
            set(favoured_instances_actual_data))
        incorrectly_detected_favoured_indices= set(favoured_instances_according_to_intervention).intersection(
            set(no_change_in_pos_instances_actual_data))
        missed_favoured_indices = set(favoured_instances_actual_data).difference(set(favoured_instances_according_to_intervention))

        print("Number of correctly detected favoured instances: " + str(len(correctly_detected_favoured_indices)))
        print("Number of incorrectly detected favoured instances: " + str(len(incorrectly_detected_favoured_indices)))
        print("Number of missed favoured instances: " + str(len(missed_favoured_indices)))
        print("________________________________________________________________")

        train_data_as_panda = pd.DataFrame(self.X_biased_train.features, columns=self.X_biased_train.feature_names)
        train_data_as_panda["romantic"] = train_data_as_panda[['romantic=no', 'romantic=yes']].idxmax(axis=1)
        train_data_as_panda["reason"] = train_data_as_panda[['reason=home', 'reason=other', 'reason=reputation', 'reason=course']].idxmax(axis=1)
        train_data_as_panda = train_data_as_panda.drop(['reason=home', 'reason=other', 'reason=reputation', 'reason=course', 'romantic=no', 'romantic=yes'], axis=1)
        print(train_data_as_panda)

        correctly_detected_favoured_instances = train_data_as_panda.loc[correctly_detected_favoured_indices]
        correctly_detected_favoured_instances["Error Label"] = "Correctly Favoured"
        incorrectly_detected_favoured_instances = train_data_as_panda.loc[
            incorrectly_detected_favoured_indices]
        incorrectly_detected_favoured_instances["Error Label"] = "Incorrectly Favoured"
        missed_favoured_instances = train_data_as_panda.loc[missed_favoured_indices]
        missed_favoured_instances["Error Label"] = "Missed Favoured"
        favouritism_data_and_errors = pd.concat([correctly_detected_favoured_instances, incorrectly_detected_favoured_instances, missed_favoured_instances], ignore_index=True, axis=0)
        self.error_analysis_subgroup_discovery(favouritism_data_and_errors, "Missed Favoured")
        self.error_analysis_subgroup_discovery(favouritism_data_and_errors, "Correctly Favoured")
        self.error_analysis_subgroup_discovery(favouritism_data_and_errors, "Incorrectly Favoured")

    def test_massaging(self):
        masseuse = Massaging()
        masseuse.fit(self.X_biased_train)
        return masseuse

    def test_situation_testing(self):
        st = Situation_Testing(k=10, threshold=0.3)  # in benchmarking study k=10, threshold=0.3
        st.fit(self.X_biased_train)

        return st

    def error_analysis_subgroup_discovery(self, error_labelled_data, target_of_interest, number_of_subgroups=15,
                                                   max_subgroup_features=4):
        """

        :param target_of_interest: {"Discriminated", "Favoured", "No Change in Positive Label", "No Change in Negative Label"}, default = None
        :param number_of_subgroups: int, default = 15
        :param max_subgroup_features: int, default = 4
        :return: None
        """
        target = ps.BinaryTarget('Error Label', target_of_interest)
        searchspace = ps.create_selectors(error_labelled_data, ignore=['Error Label'], nbins=10)
        task = ps.SubgroupDiscoveryTask(
            error_labelled_data,
            target,
            searchspace,
            result_set_size=number_of_subgroups,
            depth=max_subgroup_features,
            qf=ps.WRAccQF())
        result = ps.Apriori().execute(task)
        result_df = result.to_dataframe()
        pd.set_option("max_columns", 5)
        print(result_df[['quality', 'subgroup', 'size_sg', 'positives_sg']])
        return





def test_discrimination_detection_of_intervention_on_folds(X_biased, X_fair, categorical_attributes):
    X_biased_train, X_biased_test, X_fair_train, X_fair_test = train_test_split(X_biased, X_fair, test_size=0.2,
                                                                                random_state=42)

    biased_data_train = StandardDataset(X_biased_train, label_name="Pass", favorable_classes=[1],
                                        protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                        categorical_features=categorical_attributes)

    fair_data_train = StandardDataset(X_fair_train, label_name="Pass", favorable_classes=[1],
                                      protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                      categorical_features=categorical_attributes)

    biased_data_test = StandardDataset(X_biased_test, label_name="Pass", favorable_classes=[1],
                                       protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                       categorical_features=categorical_attributes)

    fair_data_test = StandardDataset(X_fair_test, label_name="Pass", favorable_classes=[1],
                                     protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                     categorical_features=categorical_attributes)

    tester = ErrorAnalyzer(fair_data_train, fair_data_test, biased_data_train, biased_data_test)
    tester.test_favouritism_detection_of_intervention("Massaging")


