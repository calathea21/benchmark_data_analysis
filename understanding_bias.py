import pysubgroup as ps
import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt
p9.options.figure_size = (6.5, 5.5)


def favoured_discrimination_condition(data):
    if (data["FairLabel"] == True) and (data["Pass"] == False):
        return "Discriminated"
    elif(data["FairLabel"] == False) and (data["Pass"] == True):
        return "Favoured"
    elif (data["FairLabel"] == False) and (data["Pass"] == False):
        return "No Change in Neg. Label"
    else:
        return "No Change in Pos. Label"


def add_discrimination_favoured_label(fair_data, biased_data):
    biased_data_with_disc_label = biased_data.copy()
    biased_data_with_disc_label["FairLabel"] = fair_data["Pass"]
    biased_data_with_disc_label["Discrimination_Label"] = biased_data_with_disc_label.apply(favoured_discrimination_condition, axis=1)

    return biased_data_with_disc_label

#Given a fair version of the data, as well as a biased version; this function prints some general statistics on the
#occurence of discrimination/favouritism among boys and girls
def general_statistics_favoured_vs_discriminated(fair_data, biased_data):
    biased_data_with_disc_label = add_discrimination_favoured_label(fair_data, biased_data)
    biased_boys_data = biased_data_with_disc_label[biased_data['sex'] == 'M']
    biased_girls_data = biased_data_with_disc_label[biased_data['sex'] == 'F']

    discriminated_boys = biased_boys_data[biased_boys_data['Discrimination_Label'] == "Discriminated"]
    no_change_in_neg_boys = biased_boys_data[biased_boys_data['Discrimination_Label'] == "No Change in Neg. Label"]
    no_change_in_pos_boys = biased_boys_data[biased_boys_data['Discrimination_Label'] == "No Change in Pos. Label"]
    favoured_boys = biased_boys_data[biased_boys_data['Discrimination_Label'] == "Favoured"]

    print("Number of no change in positive labels boys: " + str(len(no_change_in_pos_boys)))
    print("Number of no change in negative labels boys: " + str(len(no_change_in_neg_boys)))
    print("Number of favoured boys: " + str(len(favoured_boys)))
    print("Number of discriminated boys: " + str(len(discriminated_boys)))

    discriminated_girls = biased_girls_data[biased_girls_data['Discrimination_Label'] == "Discriminated"]
    no_change_in_neg_girls = biased_girls_data[biased_girls_data['Discrimination_Label'] == "No Change in Neg. Label"]
    no_change_in_pos_girls = biased_girls_data[biased_girls_data['Discrimination_Label'] == "No Change in Pos. Label"]
    favoured_girls = biased_girls_data[biased_girls_data['Discrimination_Label'] == "Favoured"]

    print("Number of no change in positive labels girls: " + str(len(no_change_in_pos_girls)))
    print("Number of no change in negative labels girls: " + str(len(no_change_in_neg_girls)))
    print("Number of discriminated girls: " + str(len(discriminated_girls)))
    print("Number of favoured girls: " + str(len(favoured_girls)))

    boys_actual_pass_ratio = (len(discriminated_boys) + len(no_change_in_pos_boys))/len(biased_boys_data)
    girls_actual_pass_ratio = (len(discriminated_girls) + len(no_change_in_pos_girls))/len(biased_girls_data)

    boys_biased_pass_ratio = (len(favoured_boys) + len(no_change_in_pos_boys))/ len(biased_boys_data)
    girls_biased_pass_ratio = (len(favoured_girls) + len(no_change_in_pos_girls)) / len(biased_girls_data)

    print("Difference in Positive Decision Label Ratio (Biased Data): " + str(girls_biased_pass_ratio-boys_biased_pass_ratio))
    print("Difference in Positive Decision Label Ratio (Fair Data): " + str(girls_actual_pass_ratio-boys_actual_pass_ratio))


#In this function the apriori algorithm is exectued, to find which subgroups are most associated with a target of interest
#Possible targets of interest are: "Discriminated", "Favoured", "No Change in Positive Label", "No Change in Negative Label"
def discrimination_analysis_subgroup_discovery(fair_data, biased_data, target_of_interest, number_of_subgroups=15, max_subgroup_features=4):
    """

    :param fair_data: The pandas dataset containing the fair labels, stored in the column "Pass"
    :param biased_data: The pandas dataset containing the biased labels, stored in the column "Pass"
    :param target_of_interest: {"Discriminated", "Favoured", "No Change in Positive Label", "No Change in Negative Label"}, default = None
    :param number_of_subgroups: int, default = 15
    :param max_subgroup_features: int, default = 4
    :return: None
    """
    add_discrimination_favoured_label(fair_data, biased_data)
    biased_data = biased_data.drop(["FairLabel", "Pass"], axis=1)
    target = ps.BinaryTarget('Discrimination_Label', target_of_interest)
    searchspace = ps.create_selectors(biased_data, ignore=['Discrimination_Label'], nbins=10)
    task = ps.SubgroupDiscoveryTask(
        biased_data,
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


def visualize_histogram_split_by_sex(data, var_of_interest, title):
    plot = p9.ggplot(data=data) + \
           p9.geom_bar(mapping=p9.aes(x=var_of_interest, fill="sex"), position="dodge") + \
           p9.scale_fill_manual({"F": "indianred", "M": "royalblue"}) + \
           p9.ggtitle(title)
    print(plot)


#This function plots how discrimination labels (i.e. "No change in neg. label", "No change in pos. label", "Favoured",
#"Discriminated") are distributed in specific subgroups, split by sex
def understanding_discrimination_labels_in_subgroup_split_by_sex(fair_data, biased_data, subgroup_description_dictionary, title):
    """
    :param fair_data: pandas dataframe with data and its fair labels
    :param biased_data: pandas dataframe with data and biased version of the labels
    :param subgroup_description_dictionary: dictionary giving a description of the subgroup we're interested in, of the format {1_column_name_of_interest : value_of_interest}
    :param title: string denoting which title to put for the subplot
    """
    biased_data_with_disc_label = add_discrimination_favoured_label(fair_data, biased_data)
    relevant_subgroups = biased_data_with_disc_label.loc[(biased_data_with_disc_label[list(subgroup_description_dictionary)] == pd.Series(subgroup_description_dictionary)).all(axis=1)]

    number_of_males_in_subgroup = len(relevant_subgroups[relevant_subgroups['sex'] == 'M'])
    number_of_females_in_subgroup = len(relevant_subgroups[relevant_subgroups['sex'] == 'F'])

    grouped_subgroups = relevant_subgroups.groupby('sex')["Discrimination_Label"].value_counts().to_frame(name="count").reset_index()
    grouped_subgroups.loc[grouped_subgroups['sex'] == 'M', 'percentage'] = 100 * (grouped_subgroups["count"]/number_of_males_in_subgroup)
    grouped_subgroups.loc[grouped_subgroups['sex'] == 'F', 'percentage'] = 100 * (grouped_subgroups["count"]/number_of_females_in_subgroup)
    grouped_subgroups["label"] = grouped_subgroups['count'].astype(str) + " (" + round(grouped_subgroups['percentage'], 2).astype(str) + "%)"

    plot = p9.ggplot(data=grouped_subgroups, mapping=p9.aes(x="sex", y="count", fill="Discrimination_Label")) + \
           p9.geom_bar(stat="identity") + \
           p9.geom_text(p9.aes(label="label"), position=p9.position_stack(vjust=0.5),colour="white", size=9.5) + \
           p9.ggtitle(title) + \
        p9.theme(subplots_adjust={'bottom': 0.227}) + \
           p9.theme(legend_position=(0.52, 0.1), legend_direction='horizontal', legend_text=p9.element_text(size=10),
                    legend_title=p9.element_blank(), legend_key_size=6) + \
           p9.guides(fill=p9.guide_legend(nrow=2))

    #if want to put legend in top left corner of graphlegend_position = (0.32, 0.75)
    print(plot)



