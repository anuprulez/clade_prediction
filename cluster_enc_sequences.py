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
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from scipy.stats.mstats import pearsonr


import utils

RESULT_PATH = "test_results/11_01_22/"
PATH_PRE = "data/ncov_global/"

PATH_F_DICT = PATH_PRE + "f_word_dictionaries.json"
PATH_R_DICT = PATH_PRE + "r_word_dictionaries.json"
PATH_KMER_F_DICT = PATH_PRE + "kmer_f_word_dictionaries.json"
PATH_KMER_R_DICT = PATH_PRE + "kmer_r_word_dictionaries.json"

ENC_PATH = RESULT_PATH + ""
DEC_PATH = ""


def read_data_model():

    loaded_encoder = tf.keras.models.load_model(RESULT_PATH + "pretrain_gen_encoder") #"pretrain_gen_encoder"
    loaded_decoder = tf.keras.models.load_model(RESULT_PATH + "pretrain_gen_decoder") #pretrain_gen_decoder
    file_path = RESULT_PATH + "test/20A_20B.csv"
    te_clade_files = glob.glob(file_path)
    r_dict = utils.read_json(RESULT_PATH + "r_word_dictionaries.json")
    vocab_size = len(r_dict) + 1
    total_te_loss = list()
    

    for te_name in te_clade_files:
        te_clade_df = pd.read_csv(te_name, sep="\t")
        te_X = te_clade_df["X"].drop_duplicates()
        te_y = te_clade_df["Y"].drop_duplicates()
        print(te_X)
        
    return loaded_encoder, loaded_decoder, te_X


def get_encoded_sequences(enc, dec, seq_col):
    batch_size = seq_col.shape[1]
    test_dataset_in = tf.data.Dataset.from_tensor_slices((seq_col)).batch(batch_size)
    for step, x in enumerate(test_dataset_in):
        batch_x_test = utils.pred_convert_to_array(x)
        enc_output, enc_state = enc(batch_x_test)
        return enc_state


def cluster_enc_seqs(enc_seqs):
    print(enc_seqs.shape)


def visualize_clusters(cluster_data):
    print()


if __name__ == "__main__":
    start_time = time.time()

    forward_dict = utils.read_json(PATH_F_DICT)
    rev_dict = utils.read_json(PATH_R_DICT)
    kmer_f_dict = utils.read_json(PATH_KMER_F_DICT)
    kmer_r_dict = utils.read_json(PATH_KMER_R_DICT)
   
    enc, dec, seqs = read_data_model()
    enc_seqs = get_encoded_sequences(enc, dec, seqs)
    
    cluster_data = cluster_enc_seqs(enc_seqs)

    visualize_clusters(cluster_data)
    
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
