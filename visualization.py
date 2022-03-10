import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
from math import sqrt

def plot_pass_ratio_for_each_variable_level_split_by_gender(data, variable_of_interest):
    data_var = data[variable_of_interest]
    unique_values = data_var.unique()
    unique_values = sorted(unique_values)

    female_pass_ratios = []
    male_pass_ratios = []
    female_error_bars = []
    male_error_bars = []
    for unique_value in unique_values:
        pass_and_gender_info_for_unique_value = data[data.loc[:, variable_of_interest] == unique_value][['sex', 'Pass']]

        female_pass_info_for_unique_value = pass_and_gender_info_for_unique_value[pass_and_gender_info_for_unique_value['sex'] == 'F']
        male_pass_info_for_unique_value = pass_and_gender_info_for_unique_value[pass_and_gender_info_for_unique_value['sex'] == 'M']

        number_of_females_with_value = len(female_pass_info_for_unique_value)
        print(number_of_females_with_value)
        number_of_female_passes_for_this_Value = sum(female_pass_info_for_unique_value['Pass'])
        female_pass_ratio_for_value = number_of_female_passes_for_this_Value / number_of_females_with_value
        female_pass_ratios.append(round(female_pass_ratio_for_value, 4))
        confidence = 1.96 * sqrt((female_pass_ratio_for_value * (1 - female_pass_ratio_for_value)) / number_of_females_with_value)
        female_error_bars.append(confidence)

        number_of_males_with_value = len(male_pass_info_for_unique_value)
        print(number_of_males_with_value)
        print('\n')
        number_of_male_passes_for_this_Value = sum(male_pass_info_for_unique_value['Pass'])
        male_pass_ratio_for_value = number_of_male_passes_for_this_Value / number_of_males_with_value
        male_pass_ratios.append(round(male_pass_ratio_for_value, 3))
        confidence = 1.96 * sqrt(
            (male_pass_ratio_for_value * (1 - male_pass_ratio_for_value)) / number_of_males_with_value)
        male_error_bars.append(confidence)

    x = np.arange(len(unique_values))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, male_pass_ratios, width, label='Men', yerr=male_error_bars)
    rects2 = ax.bar(x + width / 2, female_pass_ratios, width, label='Women', yerr=female_error_bars)


    ax.set_xticks(x)
    ax.set_xticklabels(unique_values)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.title("Pass Ratio for different levels of " + variable_of_interest + " split by gender")
    plt.ylabel('Pass Ratio')
    plt.xlabel(variable_of_interest)
    plt.ylim(0, 1)

    plt.show()
    return


def plot_variable_split_by_gender_and_grade(data, variable):
    possible_grades = np.arange(21)

    crosstab = pd.crosstab(data[variable], data['sex'], dropna=False)
    female_distribution = crosstab['F'].to_dict()
    male_distribution = crosstab['M'].to_dict()

    for possible_grade in possible_grades:
        if possible_grade not in female_distribution.keys():
            female_distribution[possible_grade] = 0
        if possible_grade not in male_distribution.keys():
            male_distribution[possible_grade] = 0

    male_distribution_ordered = collections.OrderedDict(sorted(male_distribution.items()))
    female_distribution_ordered = collections.OrderedDict(sorted(female_distribution.items()))

    df_male = pd.Series(male_distribution_ordered)
    df_female = pd.Series(female_distribution_ordered)

    plt.bar(possible_grades - 0.2, df_female, 0.4, label='Girls', color='red', alpha=0.5)
    plt.bar(possible_grades + 0.2, df_male, 0.4, label='Boys', color='blue', alpha=0.5)


    plt.xticks(possible_grades)
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")
    plt.title("Grade Distribution split by sex")
    plt.legend()
    plt.show()


def number_of_pass_per_block(list_of_blocks):
    number_of_pass_list = []
    for block in list_of_blocks:
        number_of_pass = block['Pass'].value_counts().loc[True]
        number_of_pass_list.append(number_of_pass)

    print(sum(number_of_pass_list))
    plt.hist(number_of_pass_list, bins=9)
    plt.show()
