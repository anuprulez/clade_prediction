import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import glob
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from scipy.stats.mstats import pearsonr

import plotly
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans


import utils

RESULT_PATH = "test_results/12_01_22/"
PATH_PRE = "data/ncov_global/"

PATH_F_DICT = PATH_PRE + "f_word_dictionaries.json"
PATH_R_DICT = PATH_PRE + "r_word_dictionaries.json"
PATH_KMER_F_DICT = PATH_PRE + "kmer_f_word_dictionaries.json"
PATH_KMER_R_DICT = PATH_PRE + "kmer_r_word_dictionaries.json"

ENC_PATH = RESULT_PATH + ""
DEC_PATH = ""
color_dict = {0: "red", 1: "green", 2: "blue"}


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
        te_X = te_clade_df["X"] #.drop_duplicates()
        te_y = te_clade_df["Y"] #.drop_duplicates()
        print(te_X)
        
    return loaded_encoder, loaded_decoder, te_y


def get_encoded_sequences(enc, dec, seq_col):
    #batch_x_test = utils.pred_convert_to_array()
    batch_size = len(seq_col)
    test_dataset_in = tf.data.Dataset.from_tensor_slices((seq_col)).batch(batch_size)
    for step, x in enumerate(test_dataset_in):
        batch_x_test = utils.pred_convert_to_array(x)
        enc_output, enc_state = enc(batch_x_test)
    return enc_state


def cluster_enc_seqs(features, orig_seq):
    print(features)
    print(enc_seqs)
    #Initialize the class object
    kmeans = KMeans(n_clusters=len(color_dict))
    #predict the labels of clusters
    cluster_labels = kmeans.fit_predict(features)
    print(cluster_labels)
    colors = list()
    x = list()
    y = list()
    for i, l in enumerate(cluster_labels):
        colors.append(int(l))
        x.append(orig_seq[i])
        y.append(l)

    scatter_df = pd.DataFrame(list(zip(x, y)), columns=["sample seq", "clusters"])

    print(scatter_df)

    scatter_df.to_csv(RESULT_PATH + "/clustered_data.csv")
    
    '''scatter_df = pd.DataFrame(list(zip(n_samples, x, y, n_sample_variants, pt_annotations, colors)), columns=["sample_name", "x", "y", "# variants", "annotations", "clusters"])
    
    scatter_df = scatter_df.sort_values(by="clusters")
    
    scatter_df.to_csv(path_plot_df)

    fig = px.scatter(scatter_df,
        x="x",
        y="y",
        color="clusters",
        hover_data=['annotations'],
        size=np.repeat(20, len(colors))
    )

    plotly.offline.plot(fig, filename='data/cluster_variants.html')

    fig.show()'''


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
    
    cluster_data = cluster_enc_seqs(enc_seqs, seqs)

    visualize_clusters(cluster_data)
    
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
