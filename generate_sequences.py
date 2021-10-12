import time
import sys
import os
import shutil

import random
import pandas as pd
import numpy as np
import logging
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

import preprocess_sequences
import utils


RESULT_PATH = "test_results/08_10_one_hot_2_CPU_20A_20B/"

min_diff = 0
max_diff = 61
enc_units = 128
LEN_AA = 1273
train_size = 1.0

clade_parent = "20B" # 20A
clade_childen = ["20I_Alpha", "20F", "20D", "21G_Lambda", "21H"] # ["20B"]

# {"20B": ["20I (Alpha, V1)", "20F", "20D", "21G (Lambda)", "21H"]}

l_dist_name = "levenshtein_distance"

generating_factor = 2

PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spikeprot0815.fasta"
PATH_SEQ_CLADE = PATH_PRE + "hcov_global.tsv"
PATH_CLADES = "data/generating_clades.json"
COMBINED_FILE = RESULT_PATH + "combined_dataframe.csv"
WUHAN_SEQ = PATH_PRE + "wuhan-hu-1-spike-prot.txt"


def read_wuhan_seq(f_dict, rev_dict):
    with open(WUHAN_SEQ, "r") as wu_file:
        wuhan_seq = wu_file.read()
        wuhan_seq = wuhan_seq.split("\n")
        wuhan_seq = "".join(wuhan_seq)
        enc_wu_seq = [str(rev_dict[item]) for item in wuhan_seq]
        return ",".join(enc_wu_seq)
        

def prepare_pred_future_seq():
    samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    clades_in_clades_out = utils.read_json(PATH_CLADES)
    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades)
    encoded_wuhan_seq = read_wuhan_seq(forward_dict, rev_dict)
    print(clades_in_clades_out)
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df, train_size=train_size, edit_threshold=max_diff)
    create_parent_child_true_seq(forward_dict, rev_dict)
    return encoded_wuhan_seq


def create_parent_child_true_seq(forward_dict, rev_dict):
    print("Loading datasets...")
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
        X_y_l = tr_clade_df[l_dist_name].tolist()
        combined_X.extend(X)
        combined_y.extend(y)
        combined_x_y_l.extend(X_y_l)
        print(len(X), len(y), len(X_y_l))
    print()
    print("train data sizes")
    print(len(combined_X), len(combined_y), len(combined_x_y_l))

    combined_dataframe = pd.DataFrame(list(zip(combined_X, combined_y, combined_x_y_l)), columns=["X", "Y", l_dist_name])
    print(combined_dataframe)

    combined_dataframe.to_csv(COMBINED_FILE, sep="\t", index=None)
    

def load_model_generated_sequences(file_path, encoded_wuhan_seq=None, gen_future=True):
    # load test data
    te_clade_files = glob.glob(file_path)
    print(te_clade_files, file_path)
    r_dict = utils.read_json(RESULT_PATH + "r_word_dictionaries.json")
    vocab_size = len(r_dict) + 1
    total_te_loss = list()
    print("Generating sequences for {}...".format(clade_parent))
    for te_name in te_clade_files:
        te_clade_df = pd.read_csv(te_name, sep="\t")
        te_X = te_clade_df["X"]
        te_y = te_clade_df["Y"]
        print(te_clade_df)
        batch_size = te_clade_df.shape[0]
        with tf.device('/device:cpu:0'):
            predict_multiple(te_X, te_y, LEN_AA, vocab_size, batch_size, encoded_wuhan_seq, gen_future)


def predict_multiple(test_x, test_y, LEN_AA, vocab_size, batch_size, encoded_wuhan_seq, gen_future):
    batch_size = test_x.shape[0]

    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)

    true_x = list()
    true_y = list()
    predicted_y = list()

    num_te_batches = int(len(test_x) / float(batch_size))

    print("Num test batches: {}".format(str(num_te_batches)))

    for step, (x, y) in enumerate(zip(test_dataset_in, test_dataset_out)):
        batch_x_test = utils.pred_convert_to_array(x)
        batch_y_test = utils.pred_convert_to_array(y)
        print(batch_x_test.shape, batch_y_test.shape)
        print("Generating multiple sequences for each test sequence...")
        for i in range(generating_factor):

            print("Generating for iter {}/{}".format(str(i+1), str(generating_factor)))
            print("Loading trained model from {}...".format(RESULT_PATH))
            loaded_encoder = tf.keras.models.load_model(RESULT_PATH + "enc_model")
            loaded_generator = tf.keras.models.load_model(RESULT_PATH + "gen_model")
            
            noise = tf.random.normal((batch_size, enc_units))

            enc_output, enc_state = loaded_encoder(batch_x_test, training=False)
            enc_state = tf.math.add(enc_state, noise)

            print(batch_x_test.shape, noise.shape, enc_state.shape)

            generated_logits = gen_step_predict(LEN_AA, batch_size, vocab_size, loaded_generator, enc_state)
            p_y = tf.math.argmax(generated_logits, axis=-1)

            one_x = utils.convert_to_string_list(batch_x_test)
            one_y = utils.convert_to_string_list(batch_y_test)
            pred_y = utils.convert_to_string_list(p_y)
 
            l_x_gen = list()
            l_b_score = list()
            l_b_wu_score = list()
            for k in range(0, len(one_x)):
               wu_bleu_score = 0.0
               l_dist_x_pred = utils.compute_Levenshtein_dist(one_x[k], pred_y[k])
               bleu_score = sentence_bleu([one_y[k].split(",")], pred_y[k].split(","))
               if not (encoded_wuhan_seq is None):
                   wu_bleu_score = sentence_bleu([encoded_wuhan_seq.split(",")], pred_y[k].split(","))
               if l_dist_x_pred > min_diff and l_dist_x_pred < max_diff:
                   l_x_gen.append(l_dist_x_pred)
                   true_x.append(one_x[k])
                   true_y.append(one_y[k])
                   predicted_y.append(pred_y[k])
                   l_b_score.append(bleu_score)
                   l_b_wu_score.append(wu_bleu_score)
            print(len(l_x_gen), l_x_gen)

            print("Step:{}, mean levenshtein distance (x and pred): {}".format(str(i+1), str(np.mean(l_x_gen))))
            print("Step:{}, median levenshtein distance (x and pred): {}".format(str(i+1), str(np.median(l_x_gen))))
            print("Step:{}, standard deviation levenshtein distance (x and pred): {}".format(str(i+1), str(np.std(l_x_gen))))
            print("Step:{}, variance levenshtein distance (x and pred): {}".format(str(i+1), str(np.var(l_x_gen))))
            print("Step:{}, mean bleu score (y and pred): {}".format(str(i+1), str(np.mean(l_b_score))))
            print("Step:{}, mean wuhan bleu score (y and pred): {}".format(str(i+1), str(np.mean(l_b_wu_score))))

            print("Generation iter {} done".format(str(i+1)))
            print("----------")
        print("Batch {} finished".format(str(step)))
        print()
    print(len(true_x), len(true_y), len(predicted_y))
    child_clades = "_".join(clade_childen)
    true_predicted_multiple = pd.DataFrame(list(zip(true_x, true_y, predicted_y)), columns=[clade_parent, child_clades, "Generated"])
    df_path = "{}true_predicted_multiple_{}_{}_{}_times.csv".format(RESULT_PATH, clade_parent, child_clades, str(generating_factor))
    true_predicted_multiple.to_csv(df_path, index=None)
    

def gen_step_predict(LEN_AA, batch_size, vocab_size, gen_decoder, dec_state):
    step_loss = tf.constant(0.0)
    pred_logits = np.zeros((batch_size, LEN_AA, vocab_size))
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(LEN_AA):
        dec_result, dec_state = gen_decoder([i_token, dec_state], training=False)
        dec_numpy = dec_result.numpy()
        pred_logits[:, t, :] = np.reshape(dec_numpy, (dec_numpy.shape[0], dec_numpy.shape[2]))
        dec_tokens = tf.math.argmax(dec_result, axis=-1)
        i_token = dec_tokens
    pred_logits = tf.convert_to_tensor(pred_logits)
    return pred_logits


if __name__ == "__main__":
    start_time = time.time()
    # enable only when predicting future sequences
    wu_seq = None
    wu_seq = prepare_pred_future_seq()
    # set gen_future = True while predicting future
    # when gen_future = False, file_path = RESULT_PATH + "test/*.csv"
    # when gen_future = True, file_path = COMBINED_FILE
    file_path = COMBINED_FILE
    #file_path = RESULT_PATH + "test/*.csv"
    load_model_generated_sequences(file_path, wu_seq, True)
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
