from understanding_bias import discrimination_analysis_subgroup_discovery, general_statistics_favoured_vs_discriminated, understanding_discrimination_labels_in_subgroup_split_by_sex
from data_preprocessing import preprocess_data, load_data_with_biased_and_unbiased_grades, add_columns_from_original_data, change_by_grade_prediction, change_by_ranking_position
from kfold_testing import test_classifier_on_folds

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #Data with its biased and unbiased grades is loaded
    data, categorical_attributes = load_data_with_biased_and_unbiased_grades()

    #Data is preprocessed and split into a fair and biased version. When preprocessing the data, the biased grades are also
    #turned into binary decision labels, according to the 'ranking strategy' (the labels of lowest two ranked individuals
    #are always changed to false, the labels of the highest two ranked individuals are always changed to true)
    X_fair, X_biased = preprocess_data(data, False, categorical_attributes, change_by_ranking_position, threshold_rank_fail=7, threshold_rank_pass=2)

    #general statistics about how biased data relates to fair data are printed
    general_statistics_favoured_vs_discriminated(X_fair, X_biased)

    #understanding discrimination for specific subgroups
    subgroup_1 = {'studytime': 1}
    subgroup_2 = {'studytime': 1, 'romantic': 'no'}
    subgroup_3 = {'Walc': 4}

    #generating plot for discrimination-label distribution (split by sex) for subgroup of students with very high alcohol consumption
    understanding_discrimination_labels_in_subgroup_split_by_sex(X_fair, X_biased,
                                                                 subgroup_3, "alcohol consumption: very high")



    #experiment to test effectiveness of fairness interventions on fair version of the labels using 10-fold cross validation
    test_classifier_on_folds(X_biased, X_fair, categorical_attributes, number_of_folds=10, test_on_fair_test_set = True, fairness_measure="Discrimination Score", performance_measure="Accuracy", title = "Tested on fair labels")

    # experiment to test effectiveness of fairness interventions on biased version of the labels using 10-fold cross validation
    test_classifier_on_folds(X_biased, X_fair, categorical_attributes, number_of_folds=10, test_on_fair_test_set = False, fairness_measure="Discrimination Score", performance_measure="Accuracy", title = "Tested on biased labels")

