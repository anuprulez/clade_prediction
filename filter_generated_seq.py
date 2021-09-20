import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import glob
import json
import matplotlib.pyplot as plt


import utils

results_path = "test_results/20A_20C_06Sept_20EPO/"
clade_parent = "20A"
clade_child = "20C"
clade_generated = "Generated"
pred_file = "true_predicted_multiple_te_x.csv"


def filter_generated_sequences():
    true_x = list()
    true_y = list()
    predicted_y = list()
    l_dist = list()
    filtered_l_dist = list()
    pred_file_path = results_path + pred_file
    pred_df = pd.read_csv(pred_file_path, sep=",")
    print(pred_df)
    print("Filtering generated sequences ...")
    for index, item in pred_df.iterrows():
        x = item[clade_parent]
        y = item[clade_child]
        gen = item[clade_generated]
        l_x_gen = utils.compute_Levenshtein_dist(x, gen)
        l_dist.append(l_x_gen)
        if l_x_gen > 0:
            true_x.append(x)
            true_y.append(y)
            predicted_y.append(gen)
            filtered_l_dist.append(l_x_gen)
    print("Mean l distance: {}".format(str(np.mean(l_dist))))
    print("Mean filtered l distance: {}".format(str(np.mean(filtered_l_dist))))
    true_predicted_multiple_filtered = pd.DataFrame(list(zip(true_x, true_y, predicted_y)), columns=[clade_parent, clade_child, "Generated"])
    df_path = "{}true_predicted_multiple_te_x_filtered.csv".format(results_path)
    true_predicted_multiple_filtered.to_csv(df_path, index=None)
    print(true_predicted_multiple_filtered)



if __name__ == "__main__":
    start_time = time.time()
    filter_generated_sequences()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
