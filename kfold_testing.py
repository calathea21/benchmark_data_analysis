from sklearn.model_selection import KFold, train_test_split
from aif360.datasets import StandardDataset
from test_classifiers import ClassifierTester
import plotnine as p9
import pandas as pd
import numpy as np
from math import sqrt
np.random.seed(4)


def test_classifier_on_train_test_split(X_biased, X_fair, categorical_attributes, test_on_fair_test_set=True):
    X_biased_train, X_biased_test, X_fair_train, X_fair_test = train_test_split(X_biased, X_fair, test_size=0.2, random_state=42)

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

    tester = ClassifierTester(fair_data_train, fair_data_test, biased_data_train, biased_data_test)
    performances = tester.test_all_algorithms(test_on_fair_test_set)
    number_of_test_samples = len(X_fair_test)
    visualize_test_set_performance_with_confidence_interval(performances, number_of_test_samples)


def test_classifier_on_folds(X_biased, X_fair, categorical_attributes, number_of_folds, test_on_fair_test_set=True, fairness_measure="Discrimination Score", performance_measure="Accuracy"):
    kf = KFold(n_splits=number_of_folds, shuffle=True)
    fold_number = 0
    all_performances = []
    for train_index, test_index in kf.split(X_biased):
        biased_data_train, fair_data_train = X_biased.iloc[train_index], X_fair.iloc[train_index]
        print(biased_data_train['sex'])
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
        performances = classifier_tester.test_all_algorithms(test_on_fair_test_set)
        performances = performances.add_prefix(str(fold_number) + '_')
        all_performances.append(performances)
        fold_number+=1

    all_performances_df = pd.concat(all_performances, axis=1)
    print(all_performances_df)
    visualize_mean_performance_and_fairness_measure(all_performances_df, fairness_measure, performance_measure)


def visualize_mean_performance_and_fairness_measure(all_performances_df, fairness_measure, performance_measure):
    all_intervention_columns = [col for col in all_performances_df.columns if col.endswith('Intervention')]
    all_performances_df['Intervention'] = all_performances_df[all_intervention_columns[0]]
    all_performances_df = all_performances_df.drop(all_intervention_columns, axis=1)
    performance_measure_columns = [col for col in all_performances_df.columns if col.endswith(performance_measure)]
    fairness_measure_columns = [col for col in all_performances_df.columns if col.endswith(fairness_measure)]
    number_of_samples = len(performance_measure_columns)

    column_name_performance = "mean_" + performance_measure
    column_name_fairness = "mean_" + fairness_measure

    all_performances_df[column_name_performance] = all_performances_df[performance_measure_columns].mean(axis=1)
    all_performances_df[column_name_fairness] = all_performances_df[fairness_measure_columns].mean(axis=1)

    all_performances_df["standard_error_performance"] = all_performances_df[performance_measure_columns].std(axis=1) / sqrt(number_of_samples)
    all_performances_df["standard_error_fairness"] = all_performances_df[fairness_measure_columns].std(axis=1) / sqrt(number_of_samples)

    all_performances_df['classifier'] = all_performances_df.index

    make_scatterplot_of_mean_performances(all_performances_df, column_name_fairness, column_name_performance)


def visualize_all_performances_kfold(all_performances_df):
    all_intervention_columns = [col for col in all_performances_df.columns if col.endswith('Intervention')]
    all_performances_df['Intervention'] = all_performances_df[all_intervention_columns[0]]
    all_performances_df = all_performances_df.drop(all_intervention_columns, axis=1)
    accuracy_columns = [col for col in all_performances_df.columns if col.endswith('Accuracy')]
    discrimination_score_columns = [col for col in all_performances_df.columns if col.endswith('Discrimination Score')]

    all_performances_df['min_accuracy'] = all_performances_df[accuracy_columns].min(axis=1)
    all_performances_df['min_discrimination_score'] = all_performances_df[discrimination_score_columns].min(axis=1)
    all_performances_df['max_accuracy'] = all_performances_df[accuracy_columns].max(axis=1)
    all_performances_df['max_discrimination_score'] = all_performances_df[discrimination_score_columns].max(axis=1)
    all_performances_df['mean_accuracy'] = all_performances_df[accuracy_columns].mean(axis=1)
    all_performances_df['mean_discrimination_score'] = all_performances_df[discrimination_score_columns].mean(axis=1)

    all_performances_df['classifier'] = all_performances_df.index

    plot_rectangles(all_performances_df)



def plot_rectangles(performances):
    performances_plot = (p9.ggplot() +
                         p9.geom_rect(data=performances,
                                      mapping=p9.aes(xmin='min_accuracy', xmax='max_accuracy',
                                                    ymin='min_discrimination_score', ymax='max_discrimination_score',
                                                    fill='Intervention'), color="black", alpha=0.5) +
                         p9.geom_text(data=performances,
                                      mapping=p9.aes(x=performances['min_accuracy']+(performances['max_accuracy']-performances['min_accuracy'])/2,
                                                    y=performances['min_discrimination_score']+(performances['max_discrimination_score']-performances['min_discrimination_score'])/2,
                                                     label='classifier'), size=8) +
                         p9.ggtitle("Accuracy and Discrimination Scores of Fairness Intervention Algorithms"))

    print(performances_plot)


def visualize_test_set_performance_with_confidence_interval(performances, n):
    performances['classifier'] = performances.index
    performances['accuracy_confidence'] = ((performances["Accuracy"] * (1 - performances["Accuracy"]))/n)**(1/2) * 1.96
    performances['upper_accuracy'] = performances["Accuracy"] + performances["accuracy_confidence"]
    performances['lower_accuracy'] = performances["Accuracy"] - performances["accuracy_confidence"]

    make_scatterplot(performances)
    return


def make_scatterplot(performances):
    performances_plot = (p9.ggplot(performances, p9.aes(x="Discrimination Score", y="Accuracy", color="Intervention")) +
                         p9.geom_point(size=4) +
                         p9.geom_errorbar(p9.aes(ymin= performances["lower_accuracy"],
                                                 ymax=performances["upper_accuracy"]), width=0.01, alpha=0.5) +
                         p9.geom_text(p9.aes(label=performances['classifier']), size=8, color='black', nudge_y=0.005))
    print(performances_plot)


def make_scatterplot_of_mean_performances(performances, fairness_measure, performance_measure):
    performances_plot = (p9.ggplot(performances, p9.aes(x=fairness_measure, y=performance_measure, color="Intervention")) +
                         p9.geom_point(size=4) +
                         p9.geom_errorbar(p9.aes(ymin= performances[performance_measure] - performances["standard_error_performance"],
                                                 ymax= performances[performance_measure] + performances["standard_error_performance"]), width=0.01, alpha=0.5) +
                         p9.geom_text(p9.aes(label=performances['classifier']), size=8, color='black', nudge_y=0.005))
    print(performances_plot)



    #p9.geom_errorbarh(p9.aes(xmin=performances[fairness_measure] - performances["standard_error_fairness"],xmax=performances[fairness_measure] + performances["standard_error_fairness"]), height=0.01, alpha=0.5)