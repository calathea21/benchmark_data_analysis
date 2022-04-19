from sklearn import tree
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from matplotlib import pyplot as plt
from apyori import apriori
import pysubgroup as ps


def learn_tree(X, y):
    clf = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy')
    clf = clf.fit(X, y)
    fig = plt.figure(figsize=(10,10))
    _ = tree.plot_tree(clf,
                       feature_names=X.columns,
                       class_names=["Discriminated", "Favoured", "No Change"],
                       filled=True)
    #plt.show()
    print(X.columns)

    print(tree.export_text(decision_tree=clf, feature_names=X.columns.values.tolist(), show_weights=True))


def get_number_of_datapoints_with_given_context(context, all_data):
    data_points_with_context = []
    for data_point in all_data:
        data_point_has_context = all(elem in data_point for elem in context)
        if data_point_has_context:
            data_points_with_context.append(data_point)
    return len(data_points_with_context)


def calc_confidence_of_just_context(context, all_data):
    number_of_points_with_context = get_number_of_datapoints_with_given_context(context, all_data)
    context_with_label = list(context)
    context_with_label.append("Label = False")
    number_of_points_with_context_and_label = get_number_of_datapoints_with_given_context(context_with_label, all_data)
    confidence = number_of_points_with_context_and_label/number_of_points_with_context
    return confidence


def check_alpha_pdcr(alpha, pd_rules_per_pd_itemset, all_data):
    alpha_discriminatory_rules = dict((pd_itemset,[]) for pd_itemset in pd_rules_per_pd_itemset)
    for pd_itemset, pd_rules in pd_rules_per_pd_itemset.items():
        for pd_rule in pd_rules:
            context = pd_rule["rule_base"] - pd_itemset
            confidence_just_context = calc_confidence_of_just_context(context, all_data)
            confidence_with_sensitive = pd_rule["confidence"]
            extended_lift = confidence_with_sensitive/confidence_just_context
            if (extended_lift > alpha):
                pd_rule["extended_lift"] = extended_lift
                alpha_discriminatory_rules[pd_itemset].append(pd_rule)
    return

def add_rule_to_corresponding_group(pd_itemsets, pd_rules, pnd_rules, rule):
    size_of_biggest_fitting_pd_itemset = 0
    biggest_fitting_pd_itemset = frozenset([])
    for pd_itemset in pd_itemsets:
        if not rule['rule_base'].isdisjoint(pd_itemset):
            intersection_itemset = rule['rule_base'].intersection(pd_itemset)
            if len(intersection_itemset) > size_of_biggest_fitting_pd_itemset:
                biggest_fitting_pd_itemset = intersection_itemset
                size_of_biggest_fitting_pd_itemset = len(intersection_itemset)

    if size_of_biggest_fitting_pd_itemset == 0:
        pnd_rules.append(rule)
    else:
        pd_rules[biggest_fitting_pd_itemset].append(rule)
    return


def extract_cr(all_rules):
    class_items = frozenset(["Discrimination_Label = Favoured", "Discrimination_Label = Discriminated"])
    #pd_itemsets = [frozenset(["sex = M"]), frozenset(["Parents_edu = 1"]), frozenset(["sex = M", "Parents_edu = 1"])]
    pd_itemsets = [frozenset(["sex = M"]), frozenset(["sex = F"])]

    #pd_rules = {frozenset(["sex = M"]): [], frozenset(["Parents_edu = 1"]): [], frozenset(["sex = M", "Parents_edu = 1"]): []}
    pd_rules = {frozenset(["sex = M"]): [], frozenset(["sex = F"]): []}
    pnd_rules = []
    for rule in all_rules:
        #check if rule contains one of the class items
        if not rule.items.isdisjoint(class_items):
            for ordering in rule.ordered_statistics:
                rule_base = ordering.items_base
                rule_consequence = ordering.items_add
                if (not rule_consequence.isdisjoint(class_items)) & (len(rule_consequence) == 1):
                    entry = {'rule_base': rule_base, 'rule_consequence': rule_consequence, 'support': rule.support, 'confidence': ordering.confidence, 'lift': ordering.lift}
                    add_rule_to_corresponding_group(pd_itemsets, pd_rules, pnd_rules, entry)

    return pd_rules, pnd_rules


def convert_to_apriori_format(X):
    list_of_dicts_format = X.to_dict('record')
    list_of_lists = []
    for dictionary in list_of_dicts_format:
        one_entry = []
        for key, value in dictionary.items():
            one_entry.append(key + " = " + str(value))
        list_of_lists.append(one_entry)
    return list_of_lists


def association_rule_mining(X, y):
    data_apriori_format = convert_to_apriori_format(X, y)
    association_rules = apriori(transactions=data_apriori_format, min_support=0.003, min_confidence=0.02, min_lift=2, min_length=2,
                                max_length=4)
    list_of_rules = list(association_rules)
    pd_rules, pnd_rules = extract_cr(list_of_rules)
    check_alpha_pdcr(2, pd_rules, data_apriori_format)
    #in ordered_statistics worden dus alle mogelijk volgordes van association rules opgeslagen
    #stel we hebben frequent itemset (X, Y) dan zijn mogelijke volgordes van rules X->Y of Y->X
    #in item_base is hetgene voor de pijl, in items_add hetgene na de pijl

def favoured_discrimination_condition(data):
    if (data["FairLabel"] == True) and (data["Pass"] == False):
        return "Discriminated"
    elif(data["FairLabel"] == False) and (data["Pass"] == True):
        return "Favoured"
    elif (data["FairLabel"] == False) and (data["Pass"] == False):
        return "No Change in Negative Label"
    else:
        return "No Change in Positive Label"


def add_discrimination_favoured_label(fair_data, biased_data):
    biased_data["FairLabel"] = fair_data["Pass"]
    biased_data["Discrimination_Label"] = biased_data.apply(favoured_discrimination_condition, axis=1)

    return biased_data


def discrimination_analysis_association_rules(fair_data, biased_data):
    add_discrimination_favoured_label(fair_data, biased_data)
    biased_data = biased_data.drop(["FairLabel", "Pass"], axis=1)
    data_apriori_format = convert_to_apriori_format(biased_data)
    association_rules = apriori(transactions=data_apriori_format, min_support=0.003, min_confidence=0.2, min_lift=5,
                                min_length=2,
                                max_length=6)
    list_of_rules = list(association_rules)
    pd_rules, pnd_rules = extract_cr(list_of_rules)
    print(pd_rules)

def discrimination_analysis_decision_tree(fair_data, biased_data, categorical_attributes):
    add_discrimination_favoured_label(fair_data, biased_data)
    categorical_attributes.append('sex')
    discrimination_labels = biased_data["Discrimination_Label"]
    biased_data = biased_data.drop(["FairLabel", "Pass", "Discrimination_Label"], axis=1)

    one_hot = pd.get_dummies(biased_data[categorical_attributes])
    biased_data = biased_data.drop(categorical_attributes, axis=1)
    biased_data = biased_data.join(one_hot)

    learn_tree(biased_data, discrimination_labels)

def discrimination_analysis_subgroup_discovery(fair_data, biased_data):
    add_discrimination_favoured_label(fair_data, biased_data)
    biased_data = biased_data.drop(["FairLabel", "Pass"], axis=1)
    target = ps.BinaryTarget('Discrimination_Label', "Favoured")
    searchspace = ps.create_selectors(biased_data, ignore=['Discrimination_Label'])
    task = ps.SubgroupDiscoveryTask(
        biased_data,
        target,
        searchspace,
        result_set_size=15,
        depth=4,
        qf=ps.WRAccQF())
    result = ps.Apriori().execute(task)
    result_df = result.to_dataframe()
    print(result_df)
    return


def general_statistics_favoured_vs_discriminated(fair_data, biased_data):
    add_discrimination_favoured_label(fair_data, biased_data)
    biased_boys_data = biased_data[biased_data['sex'] == 'M']
    biased_girls_data = biased_data[biased_data['sex'] == 'F']

    discriminated_boys = biased_boys_data[biased_boys_data['Discrimination_Label'] == "Discriminated"]
    no_change_in_neg_boys = biased_boys_data[biased_boys_data['Discrimination_Label'] == "No Change in Negative Label"]
    no_change_in_pos_boys = biased_boys_data[biased_boys_data['Discrimination_Label'] == "No Change in Positive Label"]
    favoured_boys = biased_boys_data[biased_boys_data['Discrimination_Label'] == "Favoured"]

    print("Number of no change in positive labels boys: " + str(len(no_change_in_pos_boys)))
    print("Number of no change in negative labels boys: " + str(len(no_change_in_neg_boys)))
    print("Number of favoured boys: " + str(len(favoured_boys)))
    print("Number of discriminated boys: " + str(len(discriminated_boys)))

    discriminated_girls = biased_girls_data[biased_girls_data['Discrimination_Label'] == "Discriminated"]
    no_change_in_neg_girls = biased_girls_data[biased_girls_data['Discrimination_Label'] == "No Change in Negative Label"]
    no_change_in_pos_girls = biased_girls_data[biased_girls_data['Discrimination_Label'] == "No Change in Positive Label"]
    favoured_girls = biased_girls_data[biased_girls_data['Discrimination_Label'] == "Favoured"]

    print("Number of no change in positive labels girls: " + str(len(no_change_in_pos_girls)))
    print("Number of no change in negative labels girls: " + str(len(no_change_in_neg_girls)))
    print("Number of discriminated girls" + str(len(discriminated_girls)))
    print("Number of favoured girls: " + str(len(favoured_girls)))

    boys_actual_pass_ratio = (len(discriminated_boys) + len(no_change_in_pos_boys))/len(biased_boys_data)
    girls_actual_pass_ratio = (len(discriminated_girls) + len(no_change_in_pos_girls))/len(biased_girls_data)

    boys_biased_pass_ratio = (len(favoured_boys) + len(no_change_in_pos_boys))/ len(biased_boys_data)
    girls_biased_pass_ratio = (len(favoured_girls) + len(no_change_in_pos_girls)) / len(biased_girls_data)

    print("Difference in Positive Decision Label Ratio (Biased Data): " + str(girls_biased_pass_ratio-boys_biased_pass_ratio))
    print("Difference in Positive Decision Label Ratio (Fair Data): " + str(girls_actual_pass_ratio-boys_actual_pass_ratio))
