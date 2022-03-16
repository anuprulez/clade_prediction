import os
import sys
import shutil
import itertools
from itertools import product
import json
import pandas as pd
import numpy as np
import random
from random import choices
from scipy.spatial import distance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#import tensorflow_probability as tfp
from scipy.stats.stats import pearsonr
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn import metrics
from Levenshtein import distance as lev_dist
from focal_loss import sparse_categorical_focal_loss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
import matplotlib.pyplot as plt
import torch
from torch.utils.data import WeightedRandomSampler

from focal_loss import SparseCategoricalFocalLoss
tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)

import neural_network

PATH_KMER_F_DICT = "data/ncov_global/kmer_f_word_dictionaries.json"
PATH_KMER_R_DICT = "data/ncov_global/kmer_r_word_dictionaries.json"

#m_loss = neural_network.MaskedLoss()
#bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#focal_loss_func = SparseCategoricalFocalLoss(gamma=2)

cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

#mae = tf.keras.losses.MeanAbsoluteError()
#mse = tf.keras.losses.MeanSquaredError()

test_tf_ratio = 0.0
max_norm = 50.0
enc_stddev = 1.0
dec_stddev = 0.0001
beta = 0.9999
amino_acid_codes = "QNKWFPYLMTEIARGHSDVC"


def decay_lr(lr, factor=0.95):
    return factor * lr


def decayed_learning_rate(initial_learning_rate, step, decay_rate=0.95, decay_steps=1000000):
  return initial_learning_rate * np.float_power(decay_rate, (step / decay_steps))


def loss_function(real, pred):
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
  loss = cross_entropy(y_true=real, y_pred=pred)
  loss = tf.reduce_mean(loss)
  return loss 


def make_kmers(seq, size):
    return ["".join(seq[x:x+size]) for x in range(len(seq) - size + 1)]


def compute_Levenshtein_dist(seq_in, seq_out):
    #return np.random.randint(1, 5)
    return lev_dist(seq_in, seq_out)


def clean_up(list_folders):
    for folder in list_folders:
        try:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (folder, e))
            continue


def add_padding_to_seq(seq, s_token):
    return seq #"{},{}".format(str(s_token), seq)


def get_u_kmers(x_seq, y_seq, max_l_dist, len_aa_subseq, forward_dict, start_token, unrelated):
    x_list = list()
    y_list = list()
    s_kmer = 3
    
    for i, x_i in enumerate(x_seq):
        for j, y_j in enumerate(y_seq):
            # cut sequences of specific length
            sub_x_i = x_i.split(",")[:len_aa_subseq]
            sub_x_i = ",".join(sub_x_i)
            sub_y_j = y_j.split(",")[:len_aa_subseq]
            sub_y_j = ",".join(sub_y_j)
            x_list.append(sub_x_i)
            y_list.append(sub_y_j)

    #print("Removing duplicate subsequences...")
    #x_list = list(set(x_list))
    #y_list = list(set(y_list))

    kmer_f_dict, kmer_r_dict = get_all_possible_words(amino_acid_codes, s_kmer)
    kmer_f_dict[start_token] = "<start>"
    kmer_r_dict["<start>"] = start_token

    #print(kmer_f_dict)

    save_as_json(PATH_KMER_F_DICT, kmer_f_dict)
    save_as_json(PATH_KMER_R_DICT, kmer_r_dict)

    enc_x, enc_y = encode_sequences_kmers(forward_dict, kmer_r_dict, x_list, y_list, s_kmer)
    fil_x = list()
    fil_y = list()
    l_distance = list()
    filtered_l_distance = list()
    for i, (enc_i, enc_j) in enumerate(zip(enc_x, enc_y)):
        re_x = reconstruct_seq([kmer_f_dict[int(pos)] for pos in enc_i.split(",")])
        re_y = reconstruct_seq([kmer_f_dict[int(pos)] for pos in enc_j.split(",")])
        l_dist = compute_Levenshtein_dist(re_x, re_y)
        l_distance.append(l_dist)
        if unrelated is False:
            if l_dist > 0 and l_dist < max_l_dist:
                fil_x.append(add_padding_to_seq(enc_i, start_token))
                fil_y.append(add_padding_to_seq(enc_j, start_token))
                filtered_l_distance.append(l_dist)
        else:
            if l_dist >= max_l_dist:
                fil_x.append(add_padding_to_seq(enc_i, start_token))
                fil_y.append(add_padding_to_seq(enc_j, start_token))
                filtered_l_distance.append(l_dist)
    return fil_x, fil_y, kmer_f_dict, kmer_r_dict, l_distance, filtered_l_distance


def split_test_train(x, y, split_size):
    size = int(len(x) * split_size)
    x_1, y_1 = x[0:size], y[0:size]
    x_2, y_2 = x[size:], y[size:]
    return x_1, x_2, y_1, y_2


def generate_cross_product(x_df, y_df, in_clade, out_clade, max_l_dist, len_aa_subseq, forward_dict, rev_dict, start_token, cols=["X", "Y"], unrelated=False, unrelated_threshold=15, train_nation="USA", train_pairs=True, train_size=0.8):

    print("Training: For the USA")

    X = x_df[x_df["Country"] == train_nation]
    Y = y_df[y_df["Country"] == train_nation]
    
    print(X)
    print(Y)

    print("Dropping dups in utils.generate_cross_product")
    X = X.drop_duplicates(subset=['Sequence'])
    Y = Y.drop_duplicates(subset=['Sequence'])

    X = X.sample(frac=1).reset_index(drop=True)
    Y = Y.sample(frac=1).reset_index(drop=True)

    print(X)
    print(Y)

    x_seq = X["Sequence"].tolist()
    y_seq = Y["Sequence"].tolist()

    print(len(x_seq), len(y_seq))

    '''x_split_size = int(train_size * len(x_seq))
    y_split_size = int(train_size * len(y_seq))

    x_tr_seq = x_seq[:x_split_size]
    x_te_seq = x_seq[x_split_size:]

    y_tr_seq = y_seq[:y_split_size]
    y_te_seq = y_seq[y_split_size:]

    print(len(x_tr_seq), len(y_tr_seq), len(x_te_seq), len(y_te_seq))

    print("Calculating intersection...")
    print(list(set(x_tr_seq).intersection(x_te_seq)))
    print(list(set(y_tr_seq).intersection(y_te_seq)))

    print(len(x_tr_seq), len(y_tr_seq), len(x_te_seq), len(y_te_seq))'''
    
    if unrelated is False:
        te_filename = "data/test/{}_{}.csv".format(in_clade, out_clade)
        tr_filename = "data/train/{}_{}.csv".format(in_clade, out_clade)
        val_filename = "data/validation/{}_{}.csv".format(in_clade, out_clade)
    else:
       te_filename = "data/te_unrelated/{}_{}.csv".format(in_clade, out_clade)
       tr_filename = "data/tr_unrelated/{}_{}.csv".format(in_clade, out_clade)

    #print("Filtering for range of levenshtein distance ...")
    print("Filtering for train...")
    tr_filtered_x, tr_filtered_y, kmer_f_dict, kmer_r_dict, l_distance, filtered_l_distance = get_u_kmers(x_seq, y_seq, max_l_dist, len_aa_subseq, forward_dict, start_token, unrelated)
    print("Training combination: ", len(tr_filtered_x), len(tr_filtered_y))
    print(len(filtered_l_distance), np.mean(filtered_l_distance))
    tr_filtered_dataframe = pd.DataFrame(list(zip(tr_filtered_x, tr_filtered_y)), columns=["X", "Y"])
    tr_filtered_dataframe = tr_filtered_dataframe.drop_duplicates()
    tr_filtered_dataframe = tr_filtered_dataframe.sample(frac=1).reset_index(drop=True)
    print("Combined dataframe size: {}".format(str(len(tr_filtered_dataframe.index))))
    print(tr_filtered_dataframe.drop_duplicates())
    np.savetxt("data/generated_files/tr_l_distance.txt", l_distance)
    np.savetxt("data/generated_files/tr_filtered_l_distance.txt", filtered_l_distance)
    print("Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
    print("Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))
    print("Training Filtered dataframe size: {}".format(str(len(tr_filtered_dataframe.index))))
    print()

    train_df = tr_filtered_dataframe.sample(frac=train_size, random_state=200)
    test_df = tr_filtered_dataframe.drop(train_df.index)

    print("Train dataframe:")
    print(train_df)
    print()
    print("Test dataframe:")
    print(test_df)
    print()

    '''te_x_y = list(itertools.product(x_te_seq, y_te_seq))
    print("Test combination: ", len(x_te_seq), len(y_te_seq), len(te_x_y))
    print("Filtering for range of levenshtein distance ...")
    print("Filtering for test")
    te_filtered_x, te_filtered_y, kmer_f_dict, kmer_r_dict, l_distance, filtered_l_distance = get_u_kmers(x_te_seq, y_te_seq, max_l_dist, len_aa_subseq, forward_dict, start_token, unrelated)
    print(len(filtered_l_distance), np.mean(filtered_l_distance))
    te_filtered_dataframe = pd.DataFrame(list(zip(te_filtered_x, te_filtered_y)), columns=["X", "Y"])
    #te_filtered_dataframe = te_filtered_dataframe.drop_duplicates()
    print("Combined dataframe size: {}".format(str(len(te_filtered_dataframe.index))))
    print(te_filtered_dataframe.drop_duplicates())
    np.savetxt("data/generated_files/te_l_distance.txt", l_distance)
    np.savetxt("data/generated_files/te_filtered_l_distance.txt", filtered_l_distance)
    print("Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
    print("Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))
    print("Te Filtered dataframe size: {}".format(str(len(te_filtered_dataframe.index))))'''

    train_df.to_csv(tr_filename, sep="\t", index=None)
    test_df.to_csv(te_filename, sep="\t", index=None)


    '''print("Test and validation: For rest of the world")
    x_val = x_df[x_df["Country"] != train_nation]
    y_val = y_df[y_df["Country"] != train_nation]

    print("Dropping dups in utils.generate_cross_product")
    x_val = x_val.drop_duplicates(subset=['Sequence'])
    y_val = y_val.drop_duplicates(subset=['Sequence'])

    x_val = x_val.sample(frac=1).reset_index(drop=True)
    y_val = y_val.sample(frac=1).reset_index(drop=True)

    x_val_seq = x_val["Sequence"].tolist()
    y_val_seq = y_val["Sequence"].tolist()

    # validation
    print("Filtering for validation...")
    create_dirs("data/validation")
    filtered_x_val, filtered_y_val, _, _, l_distance, filtered_l_distance = get_u_kmers(x_val_seq, y_val_seq, max_l_dist, len_aa_subseq, forward_dict, start_token, unrelated)
    filtered_dataframe_validation = pd.DataFrame(list(zip(filtered_x_val, filtered_y_val)), columns=["X", "Y"])
    filtered_dataframe_validation = filtered_dataframe_validation.drop_duplicates()
    filtered_dataframe_validation = filtered_dataframe_validation.sample(frac=1).reset_index(drop=True)
    filtered_dataframe_validation.to_csv(val_filename, sep="\t", index=None)
    filtered_dataframe_validation.to_csv(te_filename, sep="\t", index=None)
    print(len(filtered_l_distance), np.mean(filtered_l_distance))
    print("Validation Combined dataframe size: {}".format(str(len(filtered_dataframe_validation.index))))
    np.savetxt("data/generated_files/validation_l_distance.txt", l_distance)
    np.savetxt("data/generated_files/validation_filtered_l_distance.txt", filtered_l_distance)
    print("Validation Mean levenshtein dist: {}".format(str(np.mean(l_distance))))
    print("Validation Mean filtered levenshtein dist: {}".format(str(np.mean(filtered_l_distance))))
    print("Validation Filtered dataframe size: {}".format(str(len(filtered_dataframe_validation.index))))'''

    return train_df, test_df, kmer_f_dict, kmer_r_dict


def transform_noise(noise):
    shp = noise.shape
    binary = np.random.choice([0, 1], shp[0] * shp[1])
    binary_reshape = np.reshape(binary, (shp[0], shp[1]))
    noise *= binary_reshape
    return tf.convert_to_tensor(noise, dtype=tf.float32)


def get_all_possible_words(vocab, kmer_size=3):
    all_com = [''.join(c) for c in product(vocab, repeat=kmer_size)]
    kmer_f_dict = {i + 1: all_com[i] for i in range(0, len(all_com))}
    kmer_r_dict = {all_com[i]: i + 1  for i in range(0, len(all_com))}
    return kmer_f_dict, kmer_r_dict


def convert_to_array(str_data):
    tolst = str_data.tolist()
    f_list = [item.split(",") for item in tolst]
    toarray = np.array([list(map(int, lst)) for lst in f_list])
    tensor = tf.convert_to_tensor(toarray, dtype=tf.int32)
    return toarray


def pred_convert_to_array(str_data):
    tolst = str_data.numpy()
    f_list = [item.decode("utf-8").split(",") for item in tolst]
    toarray = np.array([list(map(int, lst)) for lst in f_list])
    return tf.convert_to_tensor(toarray, dtype=tf.int32)


def format_POS_variations(var_pos):
    for key in var_pos:
       var_pos[key] = np.unique(var_pos[key].tolist())
    print(var_pos[key])

  
def one_hot_encoding():
    encoded_seq = list()
    for char in sequence:
        one_hot = np.zeros(len(aa_chars))
        one_hot[int(r_word_dictionaries[char]) - 1] = 1
        encoded_seq.append(one_hot)
        
    print(encoded_seq)
    print()
    trailing_zeros = list()
    for i in range(LEN_AA - len(sequence)):
        trailing_zeros.append(np.zeros(len(aa_chars)))
    print(trailing_zeros)
    
    encoded_seq.extend(trailing_zeros)
    print()
    print(encoded_seq)
    encoded_seq = np.array(encoded_seq)
    print(encoded_seq.shape)


def read_in_out(path):
    data_df = pd.read_csv(path, sep="\t")
    samples = data_df[['Sequence_x', 'Sequence_y']]
    samples["Sequence_x"] = samples["Sequence_x"].str.split(",")
    samples["Sequence_y"] = samples["Sequence_y"].str.split(",")
    return samples


def reconstruct_seq(kmers):
    reconstructed_seq = []
    for i, km in enumerate(kmers):
         if i < len(kmers) - 1:
             reconstructed_seq.append(km[0])
         else:
             reconstructed_seq.append(km)
    return "".join(reconstructed_seq)


def encode_sequences_kmers(f_dict, kmer_r_dict, x_seq, y_seq, s_kmer):
    in_seq = list()
    out_seq = list()
    for index, (x, y) in enumerate(zip(x_seq, y_seq)):
        x = x.split(",")
        x_chars = [str(f_dict[i]) for i in x]
        x_seq = ",".join(x_chars)
        y = y.split(",")
        y_chars = [str(f_dict[i]) for i in y]
        y_seq = ",".join(y_chars)

        x_kmers = make_kmers(x_chars, s_kmer)

        encoded_x = [str(kmer_r_dict[str(i)]) for i in x_kmers]
        encoded_x = ",".join(encoded_x)
        in_seq.append(encoded_x)

        y_kmers = make_kmers(y_chars, s_kmer)
        encoded_y = [str(kmer_r_dict[str(i)]) for i in y_kmers]
        encoded_y = ",".join(encoded_y)
        out_seq.append(encoded_y)

    return in_seq, out_seq


def get_all_kmers(x_seq, y_seq, f_dict, s_kmer):
    all_kmers = list()
    for index, (x, y) in enumerate(zip(x_seq, y_seq)):
        x = x.split(",") #[1:]
        x_chars = [str(f_dict[i]) for i in x]
        x_seq = ",".join(x_chars)

        y = y.split(",") #[1:]
        y_chars = [str(f_dict[i]) for i in y]
        y_seq = ",".join(y_chars)

        x_kmers = make_kmers(x_chars, s_kmer)
        y_kmers = make_kmers(y_chars, s_kmer)

        u_x_mers = list(set(x_kmers))
        u_y_mers = list(set(y_kmers))

        all_kmers.extend(u_x_mers)
        all_kmers.extend(u_y_mers)

    return list(set(all_kmers))
        

def read_wuhan_seq(wu_path, rev_dict):
    with open(wu_path, "r") as wu_file:
        wuhan_seq = wu_file.read()
        wuhan_seq = wuhan_seq.split("\n")
        wuhan_seq = "".join(wuhan_seq)
        enc_wu_seq = [str(rev_dict[item]) for item in wuhan_seq]
        return ",".join(enc_wu_seq)


def get_words_indices(word_list):
    forward_dictionary = {i + 1: word_list[i] for i in range(0, len(word_list))}
    reverse_dictionary = {word_list[i]: i + 1  for i in range(0, len(word_list))}

    forward_dictionary["0"] = "<start>"
    reverse_dictionary["<start>"] = "0"

    return forward_dictionary, reverse_dictionary


def save_as_json(filepath, data):
    with open(filepath, 'w') as fp:
        json.dump(data, fp)


def read_json(path):
    with open(path, 'r') as fp:
        f_content = json.loads(fp.readline())
        return f_content
        

def format_clade_name(c_name):
    return c_name.replace("/", "_")
    
    
def embedding_info(dict_json):
    return len(dict_json)
    

def convert_to_string_list(l):
    try:
        l = l.numpy()
    except Exception:
        pass
    l = [",".join([str(i) for i in item]) for item in l]
    return l


def get_variation_loss(true_in, true_out, pred_logits):
    #print(true_in.shape, pred_logits.shape)
    one_hot_true_x = tf.one_hot(true_in[:, 1:], depth=pred_logits.shape[-1], axis=-1)
    one_hot_true_y = tf.one_hot(true_out[:, 1:], depth=pred_logits.shape[-1], axis=-1)

    mse_x = mse(one_hot_true_x, pred_logits)
    mse_y = mse(one_hot_true_y, pred_logits)
    return mse_y + (1.0 - mse_x)


def get_sequence_variation_percentage(true_in, pred_logits):
    seq_tokens = tf.math.argmax(pred_logits, axis=-1)
    i_tokens = convert_to_string_list(true_in)
    o_tokens = convert_to_string_list(seq_tokens)
    df_seqs = pd.DataFrame(list(zip(i_tokens, o_tokens)), columns=["X", "Pred"])
    u_df_seqs = df_seqs.drop_duplicates()
    pair_percent_variation = len(u_df_seqs.index) / float(len(df_seqs.index))
    df_x = df_seqs["X"].drop_duplicates()
    df_pred = df_seqs["Pred"].drop_duplicates()
    x_percent_variation = len(df_x.index) / float(len(df_seqs.index))
    pred_percent_variation = len(df_pred.index) / float(len(df_seqs.index))
    print("Variation scores, true x: {}, pred y: {}, x_pred pair: {}".format(str(x_percent_variation), str(pred_percent_variation), str(pair_percent_variation)))
    return x_percent_variation, pred_percent_variation, pair_percent_variation


def sample_unrelated_x_y(unrelated_X, unrelated_y, batch_size):
    un_rand_row_index = np.random.randint(0, unrelated_X.shape[0], batch_size)
    un_X = unrelated_X[un_rand_row_index]
    un_y = unrelated_y[un_rand_row_index]
    return convert_to_array(un_X), convert_to_array(un_y)


def scale_encodings(encoding):
    encoding = encoding.numpy()
    tf_enc = RobustScaler().fit_transform(encoding)
    return tf.convert_to_tensor(tf_enc, dtype=tf.float32)


def clip_weights(tensor, clip_min=-1e-2, clip_max=1e-2):
    return tf.clip_by_value(tensor, clip_value_min=clip_min, clip_value_max=clip_max)
    #return tf.clip_by_norm(tensor, clip_norm=2.0)


def create_mut_balanced_dataset(X, Y, kmer_f_dict, seq_len, batch_size):
    l_x_seq = list()
    l_y_seq = list()
    mut_pattern = dict()
    mut_pattern_dist = dict()
    for index, (x_seq, y_seq) in enumerate(zip(X, Y)):
        seq_mut_pattern = list()
        #print(x_seq, y_seq)
        x_sp = x_seq.split(",")
        y_sp = y_seq.split(",")
        x_sp = reconstruct_seq([kmer_f_dict[pos] for pos in x_sp])
        y_sp = reconstruct_seq([kmer_f_dict[pos] for pos in y_sp])

        for i, (aa_x, aa_y) in enumerate(zip(x_sp, y_sp)):
            if aa_x != aa_y:
                key = "{}>{}>{}".format(aa_x, str(i+1), aa_y)
                seq_mut_pattern.append(key)
                if key not in mut_pattern_dist:
                    mut_pattern_dist[key] = list()
                mut_pattern_dist[key].append(index)
        seq_mut_pattern = ",".join(seq_mut_pattern)
        mut_pattern[str(index)] = seq_mut_pattern
    mut_pattern_dist_freq = dict()
    for key in mut_pattern_dist:
        mut_pattern_dist_freq[key] = len(mut_pattern_dist[key])
    mut_pattern_dist_freq = {k: v for k, v in sorted(mut_pattern_dist_freq.items(), key=lambda item: item[1], reverse=True)}

    '''key_encoded = list()
    label_encoded = list()
    label_ctr = 0
    for key in mut_pattern_dist_freq:
        key_encoded.append([key for i in range(mut_pattern_dist_freq[key])])
        label_encoded.extend([label_ctr for i in range(mut_pattern_dist_freq[key])])    
        label_ctr += 1'''

    bucket_size = 30 #int(seq_len / float(batch_size))
    n_buckets = int(seq_len / float(bucket_size))
    mut_buckets = dict()
    for i in range(n_buckets):
        s_index = i*bucket_size + 1
        e_index = (i+1)*bucket_size
        bucket_range = "{}-{}".format(s_index, e_index)
        for key in mut_pattern_dist_freq:
            pos = int(key.split(">")[1])
            if pos >= s_index and pos <= e_index:
                if bucket_range not in mut_buckets:
                    mut_buckets[bucket_range] = list()
                mut_buckets[bucket_range].append(key)
    print()
    print(mut_buckets)
    print()
    for key in mut_buckets:
        print(key, len(mut_buckets[key]), mut_buckets[key])
    save_as_json("data/generated_files/mut_pattern.json", mut_pattern)
    save_as_json("data/generated_files/mut_pattern_dist.json", mut_pattern_dist)
    save_as_json("data/generated_files/mut_pattern_dist_freq.json", mut_pattern_dist_freq)
    save_as_json("data/generated_files/mut_buckets.json", mut_buckets)
    return mut_pattern, mut_pattern_dist, mut_pattern_dist_freq, mut_buckets


def find_cluster_indices(output_seqs, batch_size, datatype="train_y"):
    ## Cluster the output set of sequences and chooose sequences randomly from each cluster
    ###
    print("Clustering {}".format(datatype))
    features = convert_to_array(output_seqs)
    from sklearn.cluster import DBSCAN
    clustering_type = OPTICS(min_samples=2, min_cluster_size=2) 
    #DBSCAN(eps=0.5, min_samples=2).fit(features) #OPTICS(min_samples=2, min_cluster_size=2)
    cluster_labels = clustering_type.fit_predict(features)
    print("Number of clusters: {}".format(str(len(list(set(cluster_labels))))))
    x = list()
    y = list()
    cluster_indices_dict = dict()
    for i, l in enumerate(cluster_labels):
        x.append(output_seqs[i])
        y.append(l)
        if l not in cluster_indices_dict:
            cluster_indices_dict[l] = list()
        cluster_indices_dict[l].append(i)
    scatter_df = pd.DataFrame(list(zip(x, y)), columns=["output_seqs", "clusters"])
    scatter_df.to_csv("data/generated_files/clustered_output_seqs_data_{}.csv".format(datatype))
    return cluster_labels, cluster_indices_dict, scatter_df


def calculate_sample_weights(X_train, y_train, batch_size, pos_variations_count):
    sample_wts = list()
    weights_dict = dict()
    beta = 0.9999
    for seq_idx, seq in enumerate(y_train):
        seq_wt = 0.0
        for t, pos in enumerate(seq.split(",")):
            '''u_var_distribution = np.array(list(pos_variations_count[str(t)].values()))
            unique_cls = np.array(list(pos_variations_count[str(t)].keys()))
            all_cls = tf.repeat(unique_cls, repeats=u_var_distribution).numpy()
            random.shuffle(all_cls)
            y = all_cls
            classes = unique_cls
            le = LabelEncoder()
            y_ind = le.fit_transform(y)
            recip_freq = len(y) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
            class_wt = recip_freq[le.transform(classes)]'''

            pos_freq = pos_variations_count[str(t)]
            pos_freq = dict(pos_freq)
            
            if len(pos_freq) > 1:
                for key in pos_freq:
                    all_freq_val = list(pos_freq.values())
                    fr = pos_freq[key]
                    if str(pos) == str(key) and fr < max(all_freq_val):
                        pos_wt = (1 - beta) / (1 - beta ** fr) 
                        seq_wt += pos_wt
        sample_wts.append(seq_wt)
        weights_dict[seq_idx] = str(seq_wt)

    unrolled_x = convert_to_array(X_train)
    unrolled_y = convert_to_array(y_train)

    weighted_sampler = WeightedRandomSampler(sample_wts, len(sample_wts))

    train_dataset = torch.utils.data.TensorDataset(torch.LongTensor(unrolled_x.astype(int)), torch.LongTensor(unrolled_y.astype(int)))
    train_data_generator = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1, sampler=weighted_sampler)

    '''bal_fac = 0.0
    for index, item in enumerate(trainLoader):
        for row in item[1].numpy():
            if 6399 in row[0:15]:
                print(index, "present", row[0:15], item[1].shape)
                bal_fac += 1.0
            else:
                print(index, "not present", row[0:5], item[1].shape)
        print("-------")
    print("7285 appearance factor: {}".format(str(bal_fac / float(len(sample_wts)))))'''

    save_as_json("data/generated_files/weights_dict.json", weights_dict)
    #training_generator, steps_per_epoch = balanced_batch_generator(unrolled_x, cluster_labels, sample_weight=sample_wts, sampler=RandomUnderSampler(), batch_size=batch_size, random_state=42)
    return train_data_generator


def pairwise_dist(A, B):
    # https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)
    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
    # return pairwise euclidead difference matrix
    D = na - 2*tf.matmul(A, B, False, True) + nb
    
    D_min = tf.math.reduce_min(D)
    D_max = tf.math.reduce_max(D)

    D = (D - D_min) / ((D_max - D_min) + 1e-10)
    D = 1.0 - D
    zeros = tf.fill([A.shape[0]], 0.0)
    D = tf.linalg.set_diag(D, zeros)
    D_mean = tf.math.abs(tf.reduce_mean(D))
    D_norm = tf.math.abs(1.0 - tf.norm(D))

    return D_mean, D_norm, D


def get_pearson_coeff(enc_matr):
    pearson_coeff = list()
    for i, x in enumerate(enc_matr):
        for j, y in enumerate(enc_matr):
            pearson_coeff.append(pearsonr(x, y)[0])
    return np.mean(np.array(pearson_coeff))


def create_dirs(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def stateful_encoding(size_stateful, inputs, enc, training=False):
    stateful_batches = list()
    n_stateful_batches = int(inputs.shape[1]/float(size_stateful))
    for i in range(n_stateful_batches):
        s_batch = inputs[:, i*size_stateful: (i+1)*size_stateful]
        enc_out, enc_state = enc(s_batch, training=training)
    return enc_out, enc_state, enc


def loop_encode_decode_stateful(seq_len, batch_size, vocab_size, input_tokens, output_tokens, gen_encoder, gen_decoder, enc_units, tf_ratio, train_test, s_stateful, mut_freq, pos_variations, pos_variations_count, batch_step):
    # TODO: Implement loss wrt to WU reference genome - crossentropy(Wu - true target) - crossentropy(Wu - generated target) == 0
    # TODO: Add sample weight based on entire sample, not just per POS. Sample weight for the entire seq
    # TODO: Collect sample weight based only on changing AAs POS and not stagnant AAs POS
    loss = tf.constant(0.0)
    true_loss = tf.constant(0.0)
    global_logits = list()
    # reset state after each batch training
    enc_state_f = tf.zeros((batch_size, enc_units))
    enc_state_b = tf.zeros((batch_size, enc_units))
    n_stateful_batches = int(input_tokens.shape[1]/float(s_stateful))
    i_tokens = tf.fill([batch_size, 1], 0)
    loss_dec_loop_norm = tf.constant(0.0)
    loss_enc_state_norm = tf.constant(0.0)
    for stateful_index in range(n_stateful_batches):
        s_batch = input_tokens[:, stateful_index*s_stateful: (stateful_index+1)*s_stateful]
        enc_output, enc_state_f, enc_state_b = gen_encoder([s_batch, enc_state_f, enc_state_b], training=True)
        dec_state = tf.concat([enc_state_f, enc_state_b], -1)
        #print("Train enc norm before adding noise: ", dec_state[:, :5], tf.norm(dec_state))
        loss_enc_state_norm += tf.math.abs(max_norm - tf.norm(dec_state))
        #if stateful_index == 0:
        dec_state = tf.math.add(dec_state, tf.random.normal((dec_state.shape[0], dec_state.shape[1]), stddev=enc_stddev))
        #dec_state = tf.clip_by_norm(dec_state, clip_norm=max_norm)
        #print("Train enc norm after adding noise and clipping: ", dec_state[:, :5], tf.norm(dec_state))
        #print("Train enc norm after adding noise: ", dec_state, tf.norm(dec_state))
        #print("---")
        u_seq_len = s_batch.shape[1]
        free_run_loops = int(0.2 * u_seq_len)
        free_run_s_index = np.random.randint(0, u_seq_len - free_run_loops + 1, 1)[0]
        #print(free_run_loops, free_run_s_index)
        for t in range(s_batch.shape[1]):
            dec_result, dec_state = gen_decoder([i_tokens, dec_state], training=True)
            loss_dec_state_norm = tf.math.abs(max_norm - tf.norm(dec_state))
            #dec_state = tf.clip_by_norm(dec_state, clip_norm=max_norm)
            loss_dec_loop_norm += loss_dec_state_norm
            dec_state = tf.math.add(dec_state, tf.random.normal((dec_state.shape[0], dec_state.shape[1]), stddev=dec_stddev))
            orig_t = stateful_index * s_stateful + t
            if len(output_tokens) > 0:
                o_tokens = output_tokens[:, orig_t:orig_t+1]
                # collect different variations at each POS
                u_var_distribution = np.array(list(pos_variations_count[str(orig_t)].values()))
                unique_cls = np.array(list(pos_variations_count[str(orig_t)].keys()))
                all_cls = tf.repeat(unique_cls, repeats=u_var_distribution).numpy()
                random.shuffle(all_cls)
                y = all_cls
                classes = unique_cls
                le = LabelEncoder()
                y_ind = le.fit_transform(y)
                recip_freq = len(y) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
                class_wt = recip_freq[le.transform(classes)]

                s_wts = np.sum(class_wt)
                #print(t)
                #print(u_var_distribution)
                #print(pos_variations_count[str(orig_t)])
                #print()
                #print(class_wt)

                class_var_pos = dict()
                norm_class_var_pos = dict()
                exp_class_var_pos = dict()
                real_class_wts = list()
                for k_i, key in enumerate(unique_cls):
                    # loss input taken from paper: https://arxiv.org/pdf/1901.05555.pdf
                    class_var_pos[key] = class_wt[k_i] #/ float(s_wts)
                    norm_class_var_pos[key] = class_wt[k_i] / float(s_wts)
                    exp_class_var_pos[key] = (1 - beta) / (1 - beta ** pos_variations_count[str(orig_t)][key])
                    real_class_wts.append(exp_class_var_pos[key])

                exp_norm_u_var_distribution = np.zeros((batch_size))
                #uniform_wts = np.zeros((batch_size))

                for pos_idx, pos in enumerate(np.reshape(o_tokens, (batch_size,))):
                    exp_norm_u_var_distribution[pos_idx] = exp_class_var_pos[pos]

                '''if len(class_wt) > 1:
                    exp_norm_u_var_distribution = exp_norm_u_var_distribution / np.sum(exp_norm_u_var_distribution)'''
                #exp_norm_u_var_distribution = exp_norm_u_var_distribution / np.sum(exp_norm_u_var_distribution)
                #exp_norm_u_var_distribution = tf.convert_to_tensor(exp_norm_u_var_distribution, dtype=tf.dtypes.float32)
                #exp_norm_u_var_distribution = tf.reshape(exp_norm_u_var_distribution, (batch_size, 1))

                #print(class_var_pos)
                #print()

                '''print(exp_class_var_pos)
                print()
                print(o_tokens)
                print()
                print(exp_norm_u_var_distribution)
                print("========----=========")'''

                weighted_loss = tf.reduce_mean(cross_entropy_loss(o_tokens, dec_result, sample_weight=exp_norm_u_var_distribution))
                #weighted_loss = sparse_categorical_focal_loss(o_tokens, dec_result, gamma=5)
                #weighted_loss *= exp_norm_u_var_distribution
                #weighted_loss = tf.reduce_mean(weighted_loss)
                true_loss += tf.reduce_mean(cross_entropy_loss(o_tokens, dec_result))
                step_loss = weighted_loss
                loss += step_loss
                global_logits.append(dec_result)

            '''if t in list(range(free_run_s_index, free_run_s_index + free_run_loops)):
                #print("Free run...")
                i_tokens = tf.argmax(dec_result, axis=-1)
            else:
                #print("Fixed run...")
                i_tokens = o_tokens'''
            i_tokens = o_tokens
    #sys.exit()
    global_logits = tf.concat(global_logits, axis=-2)
    loss_dec_loop_norm = loss_dec_loop_norm / seq_len
    loss_enc_state_norm = loss_enc_state_norm / n_stateful_batches
    #loss = loss / seq_len
    #true_loss = true_loss / seq_len
    total_loss = loss #+ loss_dec_loop_norm + loss_enc_state_norm
    print("True loss: {}, Weighted loss: {}".format(str(true_loss.numpy()), str(total_loss.numpy())))
    #print("Losses: (total, true, enc norm, dec norm)", total_loss.numpy(), loss.numpy(), loss_enc_state_norm.numpy(), loss_dec_loop_norm.numpy())
    return global_logits, gen_encoder, gen_decoder, total_loss


def loop_encode_decode_predict_stateful(seq_len, batch_size, vocab_size, input_tokens, output_tokens, gen_encoder, gen_decoder, enc_units, tf_ratio, train_test, s_stateful, mut_freq): 
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
        #print("Test enc norm before adding noise: ", tf.norm(dec_state))
        dec_state = tf.math.add(dec_state, tf.random.normal((dec_state.shape[0], dec_state.shape[1]), stddev=enc_stddev))
        #dec_state = tf.clip_by_norm(dec_state, clip_norm=max_norm)
        #print("Test enc norm after adding noise and clipping: ", tf.norm(dec_state))
        #print("Test enc norm after adding noise: ", tf.norm(dec_state))
        for t in range(s_batch.shape[1]):
            orig_t = stateful_index * s_stateful + t
            dec_result, dec_state = gen_decoder([i_tokens, dec_state], training=train_test)
            #dec_state = tf.clip_by_norm(dec_state, clip_norm=max_norm)
            dec_state = tf.math.add(dec_state, tf.random.normal((dec_state.shape[0], dec_state.shape[1]), stddev=dec_stddev))
            gen_logits.append(dec_result)
            if len(output_tokens) > 0:
                o_tokens = output_tokens[:, orig_t:orig_t+1]
                step_loss = tf.reduce_mean(cross_entropy_loss(o_tokens, dec_result))
                loss += step_loss
            i_tokens = tf.argmax(dec_result, axis=-1)
    gen_logits = tf.concat(gen_logits, axis=-2)
    loss = loss / seq_len
    return gen_logits, loss


def predict_sequence(tr_epoch, tr_batch, test_dataset_in, test_dataset_out, te_batch_size, n_te_batches, seq_len, vocab_size, enc_units, loaded_encoder, loaded_generator, s_stateful, train_type, save=False):
    avg_test_loss = []
    avg_test_seq_var = []
    train_mode = False

    batch_x = list()
    batch_y = list()
    batch_pred = list()
    #for step, (batch_x_test, batch_y_test) in enumerate(zip(test_dataset_in, test_dataset_out)):
    for step in range(n_te_batches):
        batch_x_test, batch_y_test = sample_unrelated_x_y(test_dataset_in, test_dataset_out, te_batch_size)
        print("Test: true output seq:")
        print(batch_y_test[:te_batch_size,])
        # generate seqs stepwise - teacher forcing
        generated_logits, loss = loop_encode_decode_predict_stateful(seq_len, te_batch_size, vocab_size, batch_x_test, batch_y_test, loaded_encoder, loaded_generator, enc_units, test_tf_ratio, train_mode, s_stateful, dict())
        print(tf.argmax(generated_logits, axis=-1)[:te_batch_size, :])
        one_x = convert_to_string_list(batch_x_test)
        one_y = convert_to_string_list(batch_y_test)
        pred_y = convert_to_string_list(tf.math.argmax(generated_logits, axis=-1))
        batch_x.extend(one_x)
        batch_y.extend(one_y)
        batch_pred.extend(pred_y)
        variation_score = get_sequence_variation_percentage(batch_x_test, generated_logits)
        print("Test batch {} variation score: {}".format(str(step+1), str(variation_score)))
        print("Test batch {} true loss: {}".format(str(step+1), str(loss.numpy())))
        print()
        avg_test_loss.append(loss)
        avg_test_seq_var.append(variation_score)
    print()
    print("Total test seq variation in {} batches: {}".format(str(n_te_batches), str(np.mean(avg_test_seq_var))))
    print("Total test loss in {} batches: {}".format(str(n_te_batches), str(np.mean(avg_test_loss))))
    print("---")

    if save is True:
        true_predicted_multiple = pd.DataFrame(list(zip(batch_x, batch_y, batch_pred)), columns=["X", "Y", "Generated"])
        df_path = "{}{}_intermediate_training_prediction_tr_epoch_{}_tr_batch_{}_n_te_batches_{}.csv".format("data/generated_files/", train_type, str(tr_epoch+1), str(tr_batch+1), str(step+1), str(n_te_batches))
        true_predicted_multiple.to_csv(df_path, index=None)
    return np.mean(avg_test_loss), np.mean(avg_test_seq_var)


def generate_per_seq(tr_epoch, tr_batch, seq_len, te_batch_size, vocab_size, batch_x_test, batch_y_test, loaded_encoder, loaded_generator, enc_units):
    test_tf_ratio = 0.0
    train_mode = False
    s_stateful = 10
    generating_fac = 5
    #te_batch_size = 1
    batch_x = list()
    batch_pred = list()
    for index, (in_seq, out_seq) in enumerate(zip(batch_x_test, batch_y_test)):
        for i in range(generating_fac):
            print(in_seq.shape, out_seq.shape, batch_x_test.shape)
            in_seq = in_seq.reshape(1, in_seq.shape[0])
            out_seq = out_seq.reshape(1, out_seq.shape[0])
            #loop_encode_decode(seq_len, te_batch_size, vocab_size, batch_x_test, batch_y_test, loaded_encoder, loaded_generator, enc_units, test_tf_ratio, train_mode, s_stateful, dict())
            generated_logits, _, _, loss = loop_encode_decode(seq_len, te_batch_size, vocab_size, in_seq, out_seq, loaded_encoder, loaded_generator, enc_units, test_tf_ratio, train_mode, s_stateful, dict())
            print(generated_logits)
            one_x = convert_to_string_list(in_seq)
            pred_y = convert_to_string_list(tf.math.argmax(generated_logits, axis=-1))
            batch_x.append(one_x)
            batch_pred.append(pred_y)
    true_predicted_multiple = pd.DataFrame(list(zip(batch_x, batch_pred)), columns=["X", "Generated"])
    df_path = "{}true_predicted_multiple_tr_epoch_{}_tr_batch_{}_te_batch_{}_gen_fac_{}.csv".format("data/generated_files/", str(tr_epoch), str(tr_batch), str(step), str(generating_fac))
    true_predicted_multiple.to_csv(df_path, index=None)


def save_predicted_test_data(test_data_in, test_data_out, te_batch_size, enc_units, vocab_size, seq_len, size_stateful, epoch_type_name, enc_model_path, dec_model_path):
    te_encoder = tf.keras.models.load_model(enc_model_path)
    te_decoder = tf.keras.models.load_model(dec_model_path)
    test_data_in, test_data_out = convert_to_array(test_data_in), convert_to_array(test_data_out)
    n_te_batches = int(test_data_in.shape[0] / te_batch_size)
    test_tf_ratio = 0.0
    train_mode = False
    test_x = list()
    pred_y = list()
    if n_te_batches > 100:
        n_te_batches = 100
    print("Saving predicted data for test...")
    for b_c in range(n_te_batches):
        s_idx = b_c*te_batch_size
        e_idx = (b_c+1)*te_batch_size
        batch_x_test, batch_y_test = test_data_in[s_idx:e_idx, :], test_data_out[s_idx:e_idx, :]
        if batch_x_test.shape[0] == te_batch_size:
            generated_logits, loss = loop_encode_decode_predict_stateful(seq_len, te_batch_size, vocab_size, batch_x_test, batch_y_test, te_encoder, te_decoder, enc_units, test_tf_ratio, train_mode, size_stateful, dict())
            gen_tokens = tf.argmax(generated_logits, axis=-1)
            variation_score = get_sequence_variation_percentage(batch_x_test, generated_logits)
            print("Test batch {}, loss: {},  variation score: {}".format(str(b_c+1), str(loss), str(variation_score)))
            batch_x_test = convert_to_string_list(batch_x_test)
            gen_tokens = convert_to_string_list(gen_tokens)
            test_x.extend(batch_x_test)
            pred_y.extend(gen_tokens)
        else:
            break

    pred_dataframe = pd.DataFrame(list(zip(test_x, pred_y)), columns=["X", "Pred Y"])
    df_filename = "data/generated_files/true_pred_epoch_type_{}.csv".format(epoch_type_name)
    pred_dataframe.to_csv(df_filename, index=None)


def generator_step(seq_len, batch_size, vocab_size, gen_decoder, dec_state_h, dec_state_c, real_i, real_o, train_gen):
    gen_logits = list()
    step_loss = tf.constant(0.0)
    i_token = real_o[:, 0:1]
    for t in tf.range(seq_len - 1):
        new_tokens = real_o[:, t:t+2]
        dec_result, dec_state_h, dec_state_c = gen_decoder([i_token, dec_state_h, dec_state_c], training=train_gen)
        if len(real_o) > 0:
            o_token = new_tokens[:, 1:2]
            loss = cross_entropy_loss(o_token, dec_result)
            step_loss += tf.reduce_mean(loss)
        # randomly select either true output or feed generated output from previous step for forced learning
        if random.random() <= teacher_forcing_ratio:
            i_token = o_token
        else:
            i_token = tf.argmax(dec_result, axis=-1)
        gen_logits.append(dec_result)
    step_loss = step_loss / float(seq_len)
    pred_logits = tf.concat(gen_logits, axis=-2)
    return pred_logits, gen_decoder, step_loss


def generated_output_seqs(seq_len, batch_size, vocab_size, gen_decoder, dec_state_h, dec_state_c, real_i, real_o, train_gen):
    gen_logits = list()
    step_loss = tf.constant(0.0)
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(seq_len - 1):
        dec_result, dec_state_h, dec_state_c = gen_decoder([i_token, dec_state_h, dec_state_c], training=train_gen)
        gen_logits.append(dec_result)
        if len(real_o) > 0:
            o_token = real_o[:, t+1:t+2]
            loss = cross_entropy_loss(o_token, dec_result)
            step_loss += tf.reduce_mean(loss)
        #self feeding
        i_token = tf.argmax(dec_result, axis=-1)
    step_loss = step_loss / float(seq_len)
    pred_logits = tf.concat(gen_logits, axis=-2)
    gen_decoder.reset_states()
    return pred_logits, gen_decoder, step_loss


def balance_train_dataset_by_levenshtein_dist(x, y, x_y_l):
    lst_x = x
    lst_y = y
    l_dist = x_y_l.numpy()
    u_l_dist = list(set(l_dist))
    batch_size = x.shape[0]
    # create more samples, take out samples equal to the number of batch_size
    n_samples = int(batch_size / float(len(u_l_dist))) + 1
    bal_x = list()
    bal_y = list()

    for l_val in u_l_dist:
        l_val_indices = np.where(l_dist == int(l_val))
        len_indices = len(l_val_indices)
        x_rows = np.array(lst_x[l_val_indices])
        y_rows = np.array(lst_y[l_val_indices])
        rand_x_rows = np.array(choices(x_rows, k=n_samples))
        rand_y_rows = np.array(choices(y_rows, k=n_samples))

        bal_x.extend(rand_x_rows)
        bal_y.extend(rand_y_rows)
        print(l_val, len(x_rows), len(y_rows), n_samples, len(rand_x_rows), len(rand_y_rows))

    bal_x = np.array(bal_x)
    bal_y = np.array(bal_y)

    rand_idx = np.random.randint(1, bal_x.shape[0], batch_size)
 
    bal_x_bs = bal_x[rand_idx]
    bal_y_bs = bal_y[rand_idx]

    bal_x_bs = tf.convert_to_tensor(bal_x_bs, dtype=tf.int32)
    bal_y_bs = tf.convert_to_tensor(bal_y_bs, dtype=tf.int32)
    return bal_x_bs, bal_y_bs


def save_batch(batch_x, batch_y, batch_mut_distribution):
    for index, (x, y) in enumerate(zip(batch_x, batch_y)):
        true_x = x.split(",")
        true_y = y.split(",")
        for i in range(len(true_x)):
            first = true_x[i:i+1]
            sec = true_y[i:i+1]
            first_mut = first[0]
            second_mut = sec[0]
            if first_mut != second_mut:
                key = "{}>{}".format(first_mut, second_mut)
                if key not in batch_mut_distribution:
                    batch_mut_distribution[key] = 0
                batch_mut_distribution[key] += 1
    return batch_mut_distribution


def get_mutation_tr_indices(train_in, train_out, kmer_f_dict, kmer_r_dict, f_dict, r_dict, parent_child_pos_vars=dict(), parent_child_pos_vars_count=dict()):
    parent_child_mut_indices = dict()
    for index, (x, y) in enumerate(zip(train_in, train_out)):
        true_x = x.split(",")
        true_y = y.split(",")
        re_true_x = true_x
        re_true_y = true_y
        for i in range(len(true_x)):
            first = re_true_x[i:i+1][0]
            sec = re_true_y[i:i+1][0]
            if first != sec:
                key = "{}>{}>{}".format(first, (i+1), sec)
                if key not in parent_child_mut_indices:
                    parent_child_mut_indices[key] = list()
                parent_child_mut_indices[key].append(index)
            key_pos_var = "{}".format(str(i))
            if key_pos_var not in parent_child_pos_vars:
                parent_child_pos_vars[key_pos_var] = list()
                parent_child_pos_vars_count[key_pos_var] = dict()

            if int(sec) not in parent_child_pos_vars[key_pos_var]:
                parent_child_pos_vars[key_pos_var].append(int(sec))

            if int(sec) not in parent_child_pos_vars_count[key_pos_var]:
                parent_child_pos_vars_count[key_pos_var][int(sec)] = 0
            parent_child_pos_vars_count[key_pos_var][int(sec)] += 1
    save_as_json("data/generated_files/parent_child_pos_vars_{}.txt".format(str(np.random.randint(0, 1e10, 1)[0])), parent_child_pos_vars)
    save_as_json("data/generated_files/parent_child_pos_vars_count_{}.txt".format(str(np.random.randint(0, 1e10, 1)[0])), parent_child_pos_vars_count)
    return parent_child_mut_indices, parent_child_pos_vars, parent_child_pos_vars_count
