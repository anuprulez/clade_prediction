import time
import sys
import os

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

import utils

data_path = "test_results/19_10_20A_20B_unrolled_GPU/" #"test_results/19_10_20A_20B_unrolled_GPU/" # 08_10_one_hot_3_CPU_20A_20B
gen_file = "true_predicted_multiple_20B_20G_21C_Epsilon_21F_Iota_2_times_test_LD_1_4.csv"
#parent_clade = "20A"
#child_clade = "20B"

parent_clade = "20B"
child_clade = "20I_Alpha_20F_20D_21G_Lambda_21H" #"20B" #"20I_Alpha_20F_20D_21G_Lambda_21H"


def read_dataframe():

    f_dict = utils.read_json(data_path + "f_word_dictionaries.json")
    dataframe = pd.read_csv(data_path + gen_file, sep=",")
    print(dataframe)
    #parent_seqs = dataframe[parent_clade]
    #true_children_seqs = dataframe[child_clade]
    gen_seqs = dataframe["Generated"]

    check_seqs(gen_seqs)
    

def check_seqs(seqs_df):
    list_seqs = seqs_df.tolist()
    #l_dist_mat = np.zeros((len(list_seqs), len(list_seqs)))
    for i, seq in enumerate(list_seqs):
        for j, seq_y in enumerate(list_seqs):
            l_dist = utils.compute_Levenshtein_dist(seq, seq_y)
            #l_dist_mat[i, j] = l_dist
            if l_dist > 0:
                print(i, j, l_dist)
    #plt.imshow(l_dist_mat)
    #plt.show()


if __name__ == "__main__":
    start_time = time.time()
    read_dataframe()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
