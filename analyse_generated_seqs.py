import time
import sys
import os

import pandas as pd
import numpy as np
import json

import utils

data_path = "test_results/08_10_one_hot_3_CPU_20A_20B/"
gen_file = "true_predicted_multiple_20B_20I_Alpha_20F_20D_21G_Lambda_21H_2_times.csv"
parent_clade = "20B"
child_clade = "20I_Alpha_20F_20D_21G_Lambda_21H"


def read_dataframe():

    f_dict = utils.read_json(data_path + "f_word_dictionaries.json")
    dataframe = pd.read_csv(data_path + gen_file, sep=",")
    print(dataframe)
    parent_seqs = dataframe[parent_clade]
    true_children_seqs = dataframe[child_clade]
    gen_seqs = dataframe["Generated"]
    check_seqs(gen_seqs)
    

def check_seqs(seqs_df):
    list_seqs = seqs_df.tolist()
    for i, seq in enumerate(list_seqs):
        if i < len(list_seqs) - 1:
            seq_next = list_seqs[i + 1]
            l_dist = utils.compute_Levenshtein_dist(seq, seq_next)
            print(i, i+1, l_dist)

if __name__ == "__main__":
    start_time = time.time()
    read_dataframe()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
