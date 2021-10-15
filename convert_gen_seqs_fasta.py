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


def convert_seq():

    f_dict = utils.read_json(data_path + "f_word_dictionaries.json")
    dataframe = pd.read_csv(data_path + gen_file, sep=",")
    parent_seqs = dataframe[parent_clade]
    gen_seqs = dataframe[child_clade]
    convert_to_fasta(gen_seqs.tolist(), f_dict)


def convert_to_fasta(list_seqs, f_dict):
    fasta_txt = ""
    for i, seq in enumerate(list_seqs):
        fasta_txt += ">{}|Generated ".format(str(i))
        fasta_txt += "\n\n"
        letter_seq = [f_dict[item] for item in seq.split(",")]
        letter_seq = "".join(letter_seq)
        letter_seq = letter_seq + "*"
        fasta_txt += letter_seq
        fasta_txt += "\n\n"
        if i == 10:
            break
    with open(data_path + "generated_seqs.fasta", "w") as f:
        f.write(fasta_txt)


if __name__ == "__main__":
    start_time = time.time()
    convert_seq()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
