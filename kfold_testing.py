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
    performances['classifier'] = performances.index
    make_scatterplot_for_test_set_performance(performances)


def test_classifier_on_folds(X_biased, X_fair, categorical_attributes, number_of_folds, test_on_fair_test_set=True, fairness_measure="Discrimination Score", performance_measure="Accuracy", title=None):
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
    all_performances_df = prepare_to_visualize_mean_performance_and_fairness_measure(all_performances_df, fairness_measure,
                                                               performance_measure)
    make_scatterplot_of_mean_performances_over_folds(all_performances_df, "Mean " + fairness_measure, "Mean " + performance_measure, title)


def prepare_to_visualize_mean_performance_and_fairness_measure(all_performances_df, fairness_measure, performance_measure):
    all_intervention_columns = [col for col in all_performances_df.columns if col.endswith('Intervention')]
    all_performances_df['Intervention'] = all_performances_df[all_intervention_columns[0]]
    all_performances_df = all_performances_df.drop(all_intervention_columns, axis=1)
    performance_measure_columns = [col for col in all_performances_df.columns if col.endswith(performance_measure)]
    fairness_measure_columns = [col for col in all_performances_df.columns if col.endswith(fairness_measure)]
    number_of_samples = len(performance_measure_columns)

    column_name_performance = "Mean " + performance_measure
    column_name_fairness = "Mean " + fairness_measure

    all_performances_df[column_name_performance] = all_performances_df[performance_measure_columns].mean(axis=1)
    all_performances_df[column_name_fairness] = all_performances_df[fairness_measure_columns].mean(axis=1)

    all_performances_df["standard_error_performance"] = all_performances_df[performance_measure_columns].std(axis=1) / sqrt(number_of_samples)
    all_performances_df["standard_error_fairness"] = all_performances_df[fairness_measure_columns].std(axis=1) / sqrt(number_of_samples)

    all_performances_df['classifier'] = all_performances_df.index

    return all_performances_df

def make_scatterplot_for_test_set_performance(performances):
    performances_plot = (p9.ggplot(performances, p9.aes(x="Discrimination Score", y="Accuracy", color="Intervention")) +
                         p9.geom_point(size=4) +
                         p9.geom_errorbar(p9.aes(ymin= performances["lower_accuracy"],
                                                 ymax=performances["upper_accuracy"]), width=0.01, alpha=0.5) +
                         p9.geom_text(p9.aes(label=performances['classifier']), size=8, color='black', nudge_y=0.005))
    print(performances_plot)


def make_scatterplot_of_mean_performances_over_folds(performances, fairness_measure, performance_measure, title):
    performances_plot = (p9.ggplot(performances, p9.aes(x=fairness_measure, y=performance_measure, color="Intervention")) + \
                         p9.geom_point(size=4.5) + \
                         p9.geom_errorbar(p9.aes(ymin= performances[performance_measure] - performances["standard_error_performance"],
                                                 ymax= performances[performance_measure] + performances["standard_error_performance"]), width=0.015, alpha=0.7) + \
                         p9.theme(legend_position=(0.32, 0.75), legend_direction='vertical',
                                  legend_text=p9.element_text(size=12),
                                  legend_title=p9.element_blank(), legend_key_size=6.5, legend_background = p9.element_rect(color="black"),
                                  plot_title=p9.element_text(size=15.5), axis_text_y=p9.element_text(size=12), axis_text_x=p9.element_text(size=12),
                                  axis_title_y=p9.element_text(size=13.5), axis_title_x=p9.element_text(size=13.5)) + \
                         p9.scale_color_manual(values=["#6CAE75", "#B4656F", "#DDA448", "#3F88C5"]) +
                         p9.geom_text(p9.aes(label=performances['classifier']), size=9.5, color='black', nudge_y=0.005) +\
                         p9.labs(title = title) )
    print(performances_plot)
    #p9.geom_errorbarh(p9.aes(xmin=performances[fairness_measure] - performances["standard_error_fairness"],xmax=performances[fairness_measure] + performances["standard_error_fairness"]), height=0.01, alpha=0.5)