import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
plt.style.use(['science', 'no-latex', 'light'])

def scatter_plot_accuracy_vs_difference_in_positive_label(df, title):
    sbn.scatterplot(x="Accuracy", y="Discrimination Score", data=df, hue="Intervention")
    #plt.show()
    # plt.scatter(df['acc'], df['diff_in_pos'])
    for idx, row in df.iterrows():
        plt.text(row['Accuracy'], row['Discrimination Score'], idx)
    plt.title(title)
    plt.show()
