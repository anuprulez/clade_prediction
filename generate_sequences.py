import time
import sys
import os
import shutil

import random
import pandas as pd
import numpy as np
import logging
import glob
import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

import preprocess_sequences
import utils


RESULT_PATH = "test_results/11_01_22/"

min_diff = 0
max_diff = 61
train_size = 1.0
enc_units = 128
random_size = 20
LEN_AA = 16
FUTURE_GEN_TEST = "test/20A_20B.csv"

clade_parent = "20A" # 20A
clade_childen = ["20B"] #["20I_Alpha", "20F", "20D", "21G_Lambda", "21H"] 
#["20G", "21C_Epsilon", "21F_Iota"] #["20I_Alpha", "20F", "20D", "21G_Lambda", "21H"] #["20I_Alpha", "20F", "20D", "21G_Lambda", "21H"] # ["20B"]
# ["20G", "21C_Epsilon", "21F_Iota"]
# {"20B": ["20I (Alpha, V1)", "20F", "20D", "21G (Lambda)", "21H"]}

generating_factor = 50

PATH_PRE = "data/ncov_global/"
#PATH_SEQ = PATH_PRE + "spikeprot0815.fasta"
#PATH_SEQ_CLADE = PATH_PRE + "hcov_global.tsv"
PATH_SAMPLES_CLADES = PATH_PRE + "sample_clade_sequence_df.csv"
PATH_F_DICT = PATH_PRE + "f_word_dictionaries.json"
PATH_R_DICT = PATH_PRE + "r_word_dictionaries.json"
PATH_KMER_F_DICT = PATH_PRE + "kmer_f_word_dictionaries.json"
PATH_KMER_R_DICT = PATH_PRE + "kmer_r_word_dictionaries.json"
PATH_CLADES = "data/generating_clades.json"
COMBINED_FILE = RESULT_PATH + "combined_dataframe.csv"
WUHAN_SEQ = PATH_PRE + "wuhan-hu-1-spike-prot.txt"
GEN_ENC_WEIGHTS = "generator_encoder_weights.h5"
GEN_DEC_WEIGHTS = "generator_decoder_weights.h5"


def prepare_pred_future_seq():
    #samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    #print("Preprocessing sequences...")
    #encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades)
    
    clades_in_clades_out = utils.read_json(PATH_CLADES)
    print(clades_in_clades_out)
    dataf = pd.read_csv(PATH_SAMPLES_CLADES, sep=",")
    encoded_sequence_df = preprocess_sequences.filter_samples_clades(dataf)
    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df, train_size=train_size, edit_threshold=max_diff, random_size=random_size, replace=True)
    # generate only with all rows
    #create_parent_child_true_seq(forward_dict, rev_dict)
    # generate only with test rows
    create_parent_child_true_seq_test()
    return encoded_wuhan_seq
    

def create_parent_child_true_seq_test():
    print("Loading test datasets...")
    list_true_y_test = list()
    true_test_file = glob.glob(RESULT_PATH + FUTURE_GEN_TEST)
    for name in true_test_file:
        test_df = pd.read_csv(name, sep="\t")
        true_Y_test = test_df["Y"].drop_duplicates() # Corresponds to 20B for 20A - 20B training
        true_Y_test = true_Y_test.tolist()
        print(len(true_Y_test))
        list_true_y_test.extend(true_Y_test)
    print(len(list_true_y_test))

    tr_clade_files = glob.glob('data/train/*.csv')
    children_combined_y = list()
    # load train data
    print("Loading true y datasets...")
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        y = tr_clade_df["Y"].drop_duplicates()
        y = y.tolist() # Corresponds to children of 20B for 20A - 20B training
        print(len(y))
        children_combined_y.extend(y)
    print(len(list_true_y_test), len(children_combined_y))
    combined_dataframe = utils.generate_cross_product(list_true_y_test, children_combined_y, max_diff)
    combined_dataframe.to_csv(COMBINED_FILE, sep="\t", index=None)


def load_model_generated_sequences(file_path):
    # load test data
    te_clade_files = glob.glob(file_path)
    r_dict = utils.read_json(RESULT_PATH + "r_word_dictionaries.json")
    vocab_size = len(r_dict) + 1
    total_te_loss = list()
    forward_dict = utils.read_json(PATH_F_DICT)
    rev_dict = utils.read_json(PATH_R_DICT)
    kmer_f_dict = utils.read_json(PATH_KMER_F_DICT)
    kmer_r_dict = utils.read_json(PATH_KMER_R_DICT)
    encoded_wuhan_seq = utils.read_wuhan_seq(WUHAN_SEQ, rev_dict)
    print("Generating sequences for {}...".format(clade_parent))
    for te_name in te_clade_files:
        te_clade_df = pd.read_csv(te_name, sep="\t")
        te_X = te_clade_df["X"].drop_duplicates()
        te_y = te_clade_df["Y"] #.drop_duplicates()
        print(te_X)
        child_clades = "_".join(clade_childen)
        #df_true_y = pd.DataFrame(te_y, columns=[child_clades])
        true_y_df_path = "{}generated_seqs_true_y_{}.csv".format(RESULT_PATH, child_clades)
        te_y.to_csv(true_y_df_path, index=None)

        with tf.device('/device:cpu:0'):
            predict_multiple(te_X, te_y, LEN_AA, vocab_size, encoded_wuhan_seq, kmer_f_dict, kmer_r_dict)


def predict_multiple(test_x, test_y, LEN_AA, vocab_size, encoded_wuhan_seq, kmer_f_dict, kmer_r_dict):
    batch_size = 1 #test_x.shape[0]
    print(batch_size, len(test_x))
    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    #test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)
    true_x = list()
    #true_y = list()
    predicted_y = list()
    test_tf_ratio = 0.0
    size_stateful = 100
    num_te_batches = int(len(test_x) / float(batch_size))
    #num_te_batches = 50
    
    print("Num test batches: {}".format(str(num_te_batches)))
    print("Loading trained model from {}...".format(RESULT_PATH))
    loaded_encoder = tf.keras.models.load_model(RESULT_PATH + "pretrain_gen_encoder") #"pretrain_gen_encoder"
    #loaded_encoder.load_weights(RESULT_PATH + GEN_ENC_WEIGHTS)
    loaded_decoder = tf.keras.models.load_model(RESULT_PATH + "pretrain_gen_decoder") #pretrain_gen_decoder
    #loaded_decoder.load_weights(RESULT_PATH + GEN_DEC_WEIGHTS)
    for step, x in enumerate(test_dataset_in):
        batch_x_test = utils.pred_convert_to_array(x)
        #batch_y_test = utils.pred_convert_to_array(y)
        print(batch_x_test.shape)
        print("Generating sequences for the set of test sequence...")
        for i in range(generating_factor):
            l_x_gen = list()
            l_b_score = list()
            l_b_wu_score = list()
            l_ld_wuhan = list()
            print("Generating for iter {}/{}".format(str(i+1), str(generating_factor)))

            generated_logits, _, _, loss = utils.loop_encode_decode_predict(LEN_AA, batch_size, vocab_size, batch_x_test, [], loaded_encoder, loaded_decoder, enc_units, test_tf_ratio, False, size_stateful, dict())
            # compute generated sequence variation
            #batch_x_test = batch_x_test[:, 1:]
            variation_score = utils.get_sequence_variation_percentage(batch_x_test, generated_logits)
            print("Generated sequence variation score: {}".format(str(variation_score)))
            p_y = tf.math.argmax(generated_logits, axis=-1)
            
            one_x = utils.convert_to_string_list(batch_x_test)
            pred_y = utils.convert_to_string_list(p_y)
            for k in range(0, len(one_x)):
               wu_bleu_score = 0.0
               
               re_true_x = utils.reconstruct_seq([kmer_f_dict[pos] for pos in one_x[k].split(",")[1:]])
               re_pred_y = utils.reconstruct_seq([kmer_f_dict[pos] for pos in pred_y[k].split(",")])

               l_dist_x_pred = utils.compute_Levenshtein_dist(re_true_x, re_pred_y)
               #ld_wuhan_gen = utils.compute_Levenshtein_dist(encoded_wuhan_seq, pred_y[k])
               #wu_bleu_score = sentence_bleu([encoded_wuhan_seq.split(",")], pred_y[k].split(","))
               if l_dist_x_pred > min_diff and l_dist_x_pred < max_diff:
                   l_x_gen.append(l_dist_x_pred)
                   true_x.append(one_x[0])
                   predicted_y.append(pred_y[0])
                   #l_b_wu_score.append(wu_bleu_score)
                   #l_ld_wuhan.append(ld_wuhan_gen)
            print("Step:{}, mean levenshtein distance (x and pred): {}".format(str(i+1), str(np.mean(l_x_gen))))
            #print("Step:{}, mean levenshtein distance (wuhan and pred): {}".format(str(i+1), str(np.mean(l_ld_wuhan))))
            print("Step:{}, median levenshtein distance (x and pred): {}".format(str(i+1), str(np.median(l_x_gen))))
            #print("Step:{}, standard deviation levenshtein distance (x and pred): {}".format(str(i+1), str(np.std(l_x_gen))))
            #print("Step:{}, variance levenshtein distance (x and pred): {}".format(str(i+1), str(np.var(l_x_gen))))
            print("Generation iter {} done".format(str(i+1)))
            print("----------")
        print("Batch {} finished".format(str(step)))
        #if step == num_te_batches - 1:
        #    break
        print()
    print(len(true_x), len(predicted_y))
    child_clades = "_".join(clade_childen)
    true_predicted_multiple = pd.DataFrame(list(zip(true_x, predicted_y)), columns=[clade_parent, "Generated"])
    df_path = "{}generated_seqs_{}_{}_{}_times_max_LD_{}.csv".format(RESULT_PATH, clade_parent, child_clades, str(generating_factor), str(max_diff))
    true_predicted_multiple.to_csv(df_path, index=None)
    


def create_parent_child_true_seq(forward_dict, rev_dict):
    tr_clade_files = glob.glob('data/train/*.csv')
    combined_X = list()
    combined_y = list()
    combined_x_y_l = list()
    # load train data
    print("Loading datasets...")
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        X = tr_clade_df["X"].tolist()
        y = tr_clade_df["Y"].tolist()
        combined_X.extend(X)
        combined_y.extend(y)
        print(len(X), len(y))
    print()
    print("train data sizes")
    print(len(combined_X), len(combined_y))
    combined_dataframe = pd.DataFrame(list(zip(combined_X, combined_y)), columns=["X", "Y"])
    print(combined_dataframe)
    combined_dataframe.to_csv(COMBINED_FILE, sep="\t", index=None)


if __name__ == "__main__":
    start_time = time.time()
    # enable only when predicting future sequences
    #prepare_pred_future_seq()
    # when not gen_future, file_path = RESULT_PATH + "test/*.csv"
    # when gen_future, file_path = COMBINED_FILE
    #file_path = COMBINED_FILE
    file_path = RESULT_PATH + "test/20A_20B.csv"
    load_model_generated_sequences(file_path)
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
