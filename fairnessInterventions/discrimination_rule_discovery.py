from sklearn import tree
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from matplotlib import pyplot as plt
from apyori import apriori

def favoured_discrimination_condition(data):
    if (data["Pass"] == True) and (data["Predicted_Pass"] == False):
        return "Discriminated"
    elif(data["Pass"] == False) and (data["Predicted_Pass"] == True):
        return "Favoured"
    else:
        return "No Change"

def add_discrimination_favoured_label(block_info):
    block_info = block_info.replace(r'^\s*$', np.nan, regex=True)
    block_info.dropna(subset=["PredictedGrade"], inplace=True)
    block_info["Predicted_Pass"] = block_info["PredictedGrade"] >= 10

    block_info["Discrimination_Label"] = block_info.apply(favoured_discrimination_condition, axis=1)
    return block_info


def learn_tree(X, y):
    clf = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy')
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
    class_items = frozenset(["Label = False", "Label = True"])
    pd_itemsets = [frozenset(["sex = M"]), frozenset(["Parents_edu = 1"]), frozenset(["sex = M", "Parents_edu = 1"])]

    pd_rules = {frozenset(["sex = M"]): [], frozenset(["Parents_edu = 1"]): [], frozenset(["sex = M", "Parents_edu = 1"]): []}
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


def convert_to_apriori_format(X, y):
    X['Label'] = y
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


