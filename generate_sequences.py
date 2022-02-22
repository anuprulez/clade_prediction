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

#### Best results - 18_02_22_0

RESULT_PATH = "test_results/18_02_22_0/" # 04_02_22_GPU # 04_02_22_local

s_kmer = 3
#LEN_AA = 301 # It should be n - 1 (n == seq len while training)
LEN_AA = 302 # 1273 for considering entire seq length
len_aa_subseq = LEN_AA
#len_final_aa_padding = len_aa_subseq + 1
len_final_aa_padding = len_aa_subseq - s_kmer + 1
min_diff = 0
max_diff = 11 #int(LEN_AA/5)
train_size = 1.0
enc_units = 128
random_size =  450

no_models = 5
start_model_index = 16
enc_stddev = 1.0
dec_stddev = 0.0001
start_token = 0

model_type = "pre_train"
FUTURE_GEN_TEST = "test/20A_20B.csv"

clade_parent = "20A" # 20A
clade_childen = ["20B"] #["20I_Alpha", "20F", "20D", "21G_Lambda", "21H"] 
#["20G", "21C_Epsilon", "21F_Iota"] #["20I_Alpha", "20F", "20D", "21G_Lambda", "21H"] #["20I_Alpha", "20F", "20D", "21G_Lambda", "21H"] # ["20B"]
# ["20G", "21C_Epsilon", "21F_Iota"]
# {"20B": ["20I (Alpha, V1)", "20F", "20D", "21G (Lambda)", "21H"]}

generating_factor = 5

PATH_PRE = "data/ncov_global/"
#PATH_SEQ = PATH_PRE + "spikeprot0815.fasta"
#PATH_SEQ_CLADE = PATH_PRE + "hcov_global.tsv"
PATH_SAMPLES_CLADES = PATH_PRE + "sample_clade_sequence_df.csv"
PATH_F_DICT = PATH_PRE + "f_word_dictionaries.json"
PATH_R_DICT = PATH_PRE + "r_word_dictionaries.json"
PATH_KMER_F_DICT = RESULT_PATH + "kmer_f_word_dictionaries.json"
PATH_KMER_R_DICT = RESULT_PATH + "kmer_r_word_dictionaries.json"
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
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df, len_aa_subseq, start_token, train_size=train_size, edit_threshold=max_diff, random_size=random_size, replace=False, unrelated=False)
    # preprocess_sequences.make_cross_product(clades_in_clades_out, filtered_dataf, len_aa_subseq, start_token, train_size=test_train_size, edit_threshold=max_l_dist, random_size=random_clade_size, unrelated=False)
    # generate only with all rows
    #create_parent_child_true_seq(forward_dict, rev_dict)
    # generate only with test rows
    create_parent_child_true_seq_test()
    #return encoded_wuhan_seq


def generated_cross_prod(list_true_y_test, children_combined_y, max_diff, len_aa_subseq, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict):
    print("Generating cross product...")
    print(len(list_true_y_test), len(children_combined_y))
    x_y = list(itertools.product(list_true_y_test, children_combined_y))
    print(len(x_y))
    fil_x = list()
    fil_y = list()
    l_distance = list()
    filtered_l_distance = list()
    for i, (enc_i, enc_j) in enumerate(x_y):
        re_x = utils.reconstruct_seq([kmer_f_dict[pos] for pos in enc_i.split(",")[1:]])
        re_y = utils.reconstruct_seq([kmer_f_dict[pos] for pos in enc_j.split(",")[1:]])
        l_dist = utils.compute_Levenshtein_dist(re_x, re_y)
        l_distance.append(l_dist)
        if l_dist > 0 and l_dist < max_diff:
            filtered_l_distance.append(l_dist)
            fil_x.append(enc_i)
            fil_y.append(enc_j)
    filtered_dataframe = pd.DataFrame(list(zip(fil_x, fil_y)), columns=["X", "Y"])
    return filtered_dataframe


def create_parent_child_true_seq_test():
    forward_dict = utils.read_json(PATH_F_DICT)
    rev_dict = utils.read_json(PATH_R_DICT)
    kmer_f_dict = utils.read_json(PATH_KMER_F_DICT)
    kmer_r_dict = utils.read_json(PATH_KMER_R_DICT)
    #print(forward_dict)
    print("Loading test datasets...")
    list_true_y_test = list()
    true_test_file = glob.glob(RESULT_PATH + FUTURE_GEN_TEST)
    for name in true_test_file:
        test_df = pd.read_csv(name, sep="\t")
        true_Y_test = test_df["Y"].drop_duplicates() # Corresponds to 20B for 20A - 20B training
        true_Y_test = true_Y_test.tolist()
        #print(len(true_Y_test))
        list_true_y_test.extend(true_Y_test)
    print(len(list_true_y_test))

    tr_clade_files = glob.glob('data/train/*.csv')
    children_combined_y = list()

    print(tr_clade_files)
    
    # load train data
    print("Loading true y datasets...")
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        y = tr_clade_df["Y"].drop_duplicates()
        y = y.tolist() # Corresponds to children of 20B for 20A - 20B training
        print(len(y))
        children_combined_y.extend(y)
        print()
    print(len(list_true_y_test), len(children_combined_y))
    #print(list_true_y_test)
    #print()
    #print(children_combined_y)
    #combined_dataframe, _, _ = utils.generate_cross_product(list_true_y_test, children_combined_y, max_diff, len_aa_subseq, forward_dict, rev_dict, start_token, unrelated=False)
    combined_dataframe = generated_cross_prod(list_true_y_test, children_combined_y, max_diff, len_aa_subseq, forward_dict, rev_dict, kmer_f_dict, kmer_r_dict)
    # u_filtered_x_y, kmer_f_dict, kmer_r_dict = utils.generate_cross_product(u_in_clade, u_out_clade, edit_threshold, len_aa_subseq, forward_dict, rev_dict, start_token, unrelated=unrelated)
    combined_dataframe.to_csv(COMBINED_FILE, sep="\t", index=None)
    #sys.exit()


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
        te_y = te_clade_df["Y"].drop_duplicates()
        print(len(te_X), len(te_y))
        child_clades = "_".join(clade_childen)
        #df_true_y = pd.DataFrame(te_y, columns=[child_clades])
        true_y_df_path = "{}generated_seqs_true_y_{}.csv".format(RESULT_PATH, child_clades)
        te_y.to_csv(true_y_df_path, index=None)

        with tf.device('/device:cpu:0'):
            predict_multiple(te_X, te_y, len_final_aa_padding, vocab_size, encoded_wuhan_seq, forward_dict, kmer_f_dict, kmer_r_dict)


def predict_multiple(test_x, test_y, len_final_aa_padding, vocab_size, encoded_wuhan_seq, forward_dict, kmer_f_dict, kmer_r_dict):
    batch_size = 8 #test_x.shape[0]
    print(batch_size, len(test_x), len(test_y))
    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    #test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)
    true_x = list()
    #true_y = list()
    predicted_y = list()
    test_tf_ratio = 0.0
    size_stateful = 50
    num_te_batches = int(len(test_x) / float(batch_size)) 
    #num_te_batches = 1
    
    print("Num test batches: {}".format(str(num_te_batches)))

    for iter_model in range(start_model_index, start_model_index + no_models):
 
        enc_model_path = RESULT_PATH + model_type + "/" + str(iter_model) + "/enc"
        dec_model_path = RESULT_PATH + model_type + "/" + str(iter_model) + "/dec"

        print("Loading encoder from {}...".format(enc_model_path))
        print("Loading decoder from {}...".format(dec_model_path))

        loaded_encoder = tf.keras.models.load_model(enc_model_path) #"pretrain_gen_encoder" #gen_enc_model
        loaded_decoder = tf.keras.models.load_model(dec_model_path) #pretrain_gen_decoder #gen_dec_model

        for step, x in enumerate(test_dataset_in):
            batch_x_test = utils.pred_convert_to_array(x)
            if batch_x_test.shape[0] != batch_size:
                break
            #batch_y_test = utils.pred_convert_to_array(y)
            print(batch_x_test.shape)
            print("Generating sequences for the set of test sequence...")
            for i in range(generating_factor):
                l_x_gen = list()
                l_b_score = list()
                l_b_wu_score = list()
                l_ld_wuhan = list()
                print("Generating for iter {}/{}".format(str(i+1), str(generating_factor)))
                #generated_logits, _, _, loss = loop_encode_decode_predict(len_final_aa_padding, batch_size, vocab_size, batch_x_test, [], loaded_encoder, loaded_decoder, enc_units, test_tf_ratio, False, size_stateful, dict())
                generated_logits, loss = utils.loop_encode_decode_predict_stateful(len_final_aa_padding, batch_size, vocab_size, batch_x_test, [], loaded_encoder, loaded_decoder, enc_units, test_tf_ratio, False, size_stateful, dict())
                variation_score = utils.get_sequence_variation_percentage(batch_x_test, generated_logits)
                print("Generated sequence variation score: {}".format(str(variation_score)))
                p_y = tf.math.argmax(generated_logits, axis=-1)

                one_x = utils.convert_to_string_list(batch_x_test)
                pred_y = utils.convert_to_string_list(p_y)
                for k in range(0, len(one_x)):
                    wu_bleu_score = 0.0

                    re_true_x = utils.reconstruct_seq([kmer_f_dict[pos] for pos in one_x[k].split(",")[0:]])
                    re_pred_y = utils.reconstruct_seq([kmer_f_dict[pos] for pos in pred_y[k].split(",")])

                    #print(one_x[k])
                    print(re_true_x)
                    print()
                    print(re_pred_y)
                    #print(pred_y[k])
                    

                    l_dist_x_pred = utils.compute_Levenshtein_dist(re_true_x, re_pred_y)
                    
                    print(l_dist_x_pred)
                    print("----------")
                    #ld_wuhan_gen = utils.compute_Levenshtein_dist(encoded_wuhan_seq, pred_y[k])
                    #wu_bleu_score = sentence_bleu([encoded_wuhan_seq.split(",")], pred_y[k].split(","))
                    if l_dist_x_pred > min_diff and l_dist_x_pred < max_diff:
                        l_x_gen.append(l_dist_x_pred)
                        true_x.append(one_x[k])
                        predicted_y.append(pred_y[k])
                        #l_b_wu_score.append(wu_bleu_score)
                        #l_ld_wuhan.append(ld_wuhan_gen)
                print(len(true_x), len(predicted_y))
                print("Step:{}, mean levenshtein distance (x and pred): {}".format(str(i+1), str(np.mean(l_x_gen))))
                #print("Step:{}, mean levenshtein distance (wuhan and pred): {}".format(str(i+1), str(np.mean(l_ld_wuhan))))
                print("Step:{}, median levenshtein distance (x and pred): {}".format(str(i+1), str(np.median(l_x_gen))))
                #print("Step:{}, standard deviation levenshtein distance (x and pred): {}".format(str(i+1), str(np.std(l_x_gen))))
                #print("Step:{}, variance levenshtein distance (x and pred): {}".format(str(i+1), str(np.var(l_x_gen))))
                print("Generation iter {} done".format(str(i+1)))
                print("----------")
            print("Batch {} finished".format(str(step)))
            print()
    print(len(true_x), len(predicted_y))
    child_clades = "_".join(clade_childen)
    true_predicted_multiple = pd.DataFrame(list(zip(true_x, predicted_y)), columns=[clade_parent, "Generated"])
    utils.create_dirs(RESULT_PATH + "model_generated_sequences")
    df_path = "{}model_generated_sequences/generated_seqs_{}_{}_{}_{}.csv".format(RESULT_PATH, clade_parent, child_clades, str(np.random.randint(0, 2000000, 1)[0]), model_type)
    true_predicted_multiple.to_csv(df_path, index=None)


'''def loop_encode_decode_predict_stateful(seq_len, batch_size, vocab_size, input_tokens, output_tokens, gen_encoder, gen_decoder, enc_units, tf_ratio, train_test, s_stateful, mut_freq): 
    enc_state_f = tf.zeros((batch_size, enc_units))
    enc_state_b = tf.zeros((batch_size, enc_units))
    n_stateful_batches = int(input_tokens.shape[1]/float(s_stateful))
    i_tokens = tf.fill([batch_size, 1], 0)
    gen_logits = list()
    loss = tf.constant(0.0)
    for stateful_index in range(n_stateful_batches):
        s_batch = input_tokens[:, stateful_index*s_stateful: (stateful_index+1)*s_stateful]
        enc_output, enc_state_f, enc_state_b = gen_encoder([s_batch, enc_state_f, enc_state_b], training=train_test)
        dec_state = tf.concat([enc_state_f, enc_state_b], -1)
        dec_state = tf.math.add(dec_state, tf.random.normal((dec_state.shape[0], dec_state.shape[1]), stddev=enc_stddev))
        for t in range(s_batch.shape[1]):
            orig_t = stateful_index * s_stateful + t
            dec_result, dec_state = gen_decoder([i_tokens, dec_state], training=train_test)
            #dec_state = tf.math.add(dec_state, tf.random.normal((dec_state.shape[0], dec_state.shape[1]), stddev=dec_stddev))
            gen_logits.append(dec_result)
            if len(output_tokens) > 0:
                o_tokens = output_tokens[:, orig_t:orig_t+1]
                step_loss = tf.reduce_mean(cross_entropy_loss(o_tokens, dec_result))
                loss += step_loss
            i_tokens = tf.argmax(dec_result, axis=-1)
    gen_logits = tf.concat(gen_logits, axis=-2)
    loss = loss / seq_len
    return gen_logits, loss'''


def loop_encode_decode_predict(seq_len, batch_size, vocab_size, input_tokens, output_tokens, gen_encoder, gen_decoder, enc_units, tf_ratio, train_test, s_stateful, mut_freq): 
    show = 2
    enc_output, enc_state = gen_encoder(input_tokens) #, training=False
    enc_norm = tf.norm(enc_state)
    dec_state = enc_state
    #print(dec_state)
    #print()
    dec_state = tf.math.add(dec_state, tf.random.normal((dec_state.shape[0], dec_state.shape[1]), stddev=enc_stddev))
    #print(dec_state[:show, :])
    loss = tf.constant(0.0)
    gen_logits = list()
    o_state_norm = list()
    i_tokens = tf.fill([batch_size, 1], 0)
    for t in range(seq_len - 1):
        dec_result, dec_state = gen_decoder([i_tokens, dec_state]) #, training=False
        gen_logits.append(dec_result)
        o_state_norm.append(tf.norm(dec_state))
        #dec_state = tf.math.add(dec_state, tf.random.normal((dec_state.shape[0], dec_state.shape[1]), stddev=dec_stddev))
        if len(output_tokens) > 0:
            o_tokens = output_tokens[:, t+1:t+2]
            step_loss = tf.reduce_mean(cross_entropy_loss(o_tokens, dec_result))
            loss += step_loss
        i_tokens = tf.argmax(dec_result, axis=-1)
        '''temp = 0.99
        dec_result_temp = tf.math.log(dec_result / temp)
        dec_result_temp = tf.math.exp(dec_result_temp) #/ float(dec_result.shape[-1])
        #topk_sce = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
        topk_i_tokens = tf.math.top_k(dec_result, k=5)
        topk_i_tokens_temp = tf.math.top_k(dec_result_temp, k=5)
        #print(o_tokens)
        #print(dec_result.shape)
        print(t, topk_i_tokens)'''
        #print(t, topk_i_tokens_temp)
        #print("------------")
    gen_logits = tf.concat(gen_logits, axis=-2)
    loss = loss / seq_len
    print("Encoder norm: {}".format(str(tf.norm(enc_state))))
    print("Decoder norm: {}".format(str(np.mean(o_state_norm))))
    #print("--------------")
    return gen_logits, gen_encoder, gen_decoder, loss

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
