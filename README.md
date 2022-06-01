# benchmark_data_analysis
Work on new ``Portuguese Students" dataset with biased and fair version of the label

The code in this repository was used to generate the experimental results for the following paper:

Lenders, D. & Calders, T. (2022). Performance vs. Predicted Performance - A New Benchmarking Dataset for Fair ML. _NeurIPS 2021 Datasets and Benchmarks Track_. [submitted for publication]

Experiments include:

- Benchmarking experiment to test the effectiveness of different fairness interventions on our new dataset (random seeds, and hyperparameters of the algorithms are as seen in the code)
- Apriori subgroup discovery, to better understand which subgroups are affected by discrimination in our data
