import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def load_data_with_biased_and_unbiased_grades():
    data = pd.read_excel("PortugueseStudentsWithBiasedLabels.xlsx")
    data = data.set_index("index")

    categorical_attributes = ["romantic", "reason"]
    return data, categorical_attributes

def change_by_grade_prediction(data, threshold_grade):
    data["Predicted_Pass"] = data["PredictedGrade"] >= threshold_grade
    return


def change_by_ranking_position(data, threshold_rank_fail, threshold_rank_pass):
    data["Predicted_Pass"] = np.where(((data['Pass'] == True) & (data['PredictedRank'] < threshold_rank_fail) | (data['Pass'] == False) & (data['PredictedRank'] <= threshold_rank_pass)), True, False)
    return


def preprocess_data(data, one_hot_encoding, categorical_attributes, changing_function, **kwargs):
    """

    :param data: pandas Dataframe - Data containing the biased and unbiased version of the grades
    :param one_hot_encoding: boolean - Denotes whether one-hot-encoding should be part of the preprocessing or not
    :param categorical_attributes: list of string - List of all the categorical attribute names that will be one-hot-encoded if specified
    :param changing_function: function - name of the function used to obtain the biased binary decision labels from the biased grade predictions
    :param kwargs: arguments that are passed to the changing_function
    :return: pandas dataframe, pandas datframe - dataframe containing the fair version of the labels, dataframe containing the biased version of the labelste
    """
    data.dropna(subset=["PredictedGrade"], inplace=True)

    changing_function(data, **kwargs)

    data = data.drop(["ParticipantID", "name", "G3", "PredictedRank", "PredictedGrade", "StereotypeActivation"], axis=1) #"StereotypeActivation"

    if one_hot_encoding:
        one_hot = pd.get_dummies(data[categorical_attributes])
        data = data.drop(categorical_attributes, axis=1)
        data = data.join(one_hot)

    fair_data = data.copy()
    fair_data = fair_data.drop(["Predicted_Pass"], axis=1)

    biased_data = data.drop(["Pass"], axis=1)
    biased_data = biased_data.rename(columns={"Predicted_Pass": "Pass"})

    return fair_data, biased_data



def add_columns_from_original_data(columns_to_add, data_with_biased_and_fair_grades, categorical_attributes):
    """

    :param columns_to_add: list of string - Column names from original dataset to add to the benchmark data
    :param data_with_biased_and_fair_grades: pandas dataframe - Datframe containing the biased and fair version of the students' grades
    :param categorical_attributes: list of string - List of the names of the categorical attributes as found in data_with_biased_and_fair_grades
    :return: pandas Dataframe, list of string - Data with the columns from the original dataset added, list of all the categorical attributes that are now in this data
    """
    categorical_attributes_org_data = {"school", "address", "famsize", "Pstatus", "Mjob", "Fjob", "guardian", "schoolsup",
                                       "famsup", "paid", "activities", "nursery", "higher", "internet", "subject"}
    original_data = pd.read_excel("original_data.xlsx")
    original_data_columns_to_add = original_data[columns_to_add].loc[data_with_biased_and_fair_grades.index]
    data_with_biased_and_fair_grades[columns_to_add] = original_data_columns_to_add

    categorical_attributes_added_to_data = categorical_attributes_org_data.intersection(set(columns_to_add))
    categorical_attributes.extend(list(categorical_attributes_added_to_data))

    return data_with_biased_and_fair_grades, categorical_attributes


