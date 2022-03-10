import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

def load_data_with_biased_and_unbiased_grades():
    data = pd.read_excel("PortugueseStudentsWithBiasedLabels.xlsx")
    data = data.set_index("index")
    return data

def preprocess_data(data, change_by_fail_pred, one_hot_encoding=True):
    data = data.replace(r'^\s*$', np.nan, regex=True)
    data.dropna(subset=["PredictedGrade"], inplace=True)

    if change_by_fail_pred:
        data["Predicted_Pass"] = data["PredictedGrade"] >= 10

    else:
        data["Predicted_Pass"] = np.where(((data['Pass'] == True) & (data['PredictedRank'] < 7) | (data['Pass'] == False) & (data['PredictedRank'] < 3)), True, False)

    data = data.drop(["ParticipantID", "name", "G3", "PredictedRank", "StereotypeActivation", "PredictedGrade"], axis=1)
    categorical_attributes = ['romantic', 'reason']

    if one_hot_encoding:
        one_hot = pd.get_dummies(data[categorical_attributes])
        data = data.drop(categorical_attributes, axis=1)
        data = data.join(one_hot)

    fair_data = data.copy()
    fair_data = fair_data.drop(["Predicted_Pass"], axis=1)

    data = data.drop(["Pass"], axis=1)
    data = data.rename(columns={"Predicted_Pass": "Pass"})

    return data, fair_data, categorical_attributes


def add_columns_from_original_data(columns_to_add, qualtrics_data):
    original_data = pd.read_excel("original_data.xlsx")
    original_data_columns_to_add = original_data[columns_to_add].loc[qualtrics_data.index]
    qualtrics_data[columns_to_add] = original_data_columns_to_add
    return


def train_test_toStandardDataset(X_biased, X_fair, categorical_attributes):
    X_biased_train, X_biased_test, X_fair_train, X_fair_test = train_test_split(X_biased, X_fair, test_size=0.2, random_state=42)

    portuguese_class_train_data_biased = StandardDataset(X_biased_train, label_name="Pass", favorable_classes=[1],
                                                         protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                                         categorical_features=categorical_attributes)
    portuguese_class_test_data_biased = StandardDataset(X_biased_test, label_name="Pass", favorable_classes=[1],
                                                  protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                                  categorical_features=categorical_attributes)

    portuguese_class_train_data_fair = StandardDataset(X_fair_train, label_name="Pass", favorable_classes=[1],
                                                       protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                                       categorical_features=categorical_attributes)

    portuguese_class_test_data_fair = StandardDataset(X_fair_test, label_name="Pass", favorable_classes=[1],
                                                        protected_attribute_names=['sex'], privileged_classes=[["F"]],
                                                        categorical_features=categorical_attributes)

    return portuguese_class_train_data_biased, portuguese_class_test_data_biased, portuguese_class_train_data_fair, \
           portuguese_class_test_data_fair


