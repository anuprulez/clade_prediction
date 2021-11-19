import time
import sys
import os

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

import utils

data_path = "test_results/11_11_local/" #"test_results/19_10_20A_20B_unrolled_GPU/" # 08_10_one_hot_3_CPU_20A_20B
gen_file = "true_predicted_multiple_20A_2_0_B_5_times_max_LD_2000.csv"
#parent_clade = "20A"
#child_clade = "20B"

parent_clade = "20B"
child_clade = "20I_Alpha_20F_20D_21G_Lambda_21H" #"20B" #"20I_Alpha_20F_20D_21G_Lambda_21H"


def read_wuhan_hu_1_spike(r_dict):
    wuhan_seq = ""
    path = data_path + "wuhan-hu-1-spike-prot.txt"
    with open(path, "r") as f_seq:
        original_seq = f_seq.read()
    wuhan_seq = original_seq.split("\n")
    wuhan_seq = wuhan_seq[:len(wuhan_seq) - 1]
    wuhan_seq = "".join(wuhan_seq)
    enc_wuhan_seq = [str(r_dict[aa]) for aa in wuhan_seq]
    return ",".join(enc_wuhan_seq)


def read_dataframe():

    f_dict = utils.read_json(data_path + "f_word_dictionaries.json")
    r_dict = utils.read_json(data_path + "r_word_dictionaries.json")
    dataframe = pd.read_csv(data_path + gen_file, sep=",")
    print(dataframe)
    #parent_seqs = dataframe[parent_clade]
    #true_children_seqs = dataframe[child_clade]
    gen_seqs = dataframe["Generated"]
    print(gen_seqs)
    u_gen_seqs = gen_seqs.drop_duplicates()
    print()
    print(u_gen_seqs)
    '''u_gen_seqs = u_gen_seqs.tolist()
    u_gen_seqs = u_gen_seqs[0].split(",")
    #print(u_gen_seqs)

    enc_original_wuhan_seq = read_wuhan_hu_1_spike(r_dict)

    original_wuhan = enc_original_wuhan_seq.split(",")
    #print(original_wuhan)

    for index, aa in enumerate(zip(original_wuhan, u_gen_seqs)):
        o_aa, gen_aa = aa[0], aa[1]
        if o_aa != gen_aa:
            print(index+1, f_dict[o_aa], f_dict[gen_aa])
            print()  
    #check_seqs(gen_seqs)'''
    

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
