import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import glob
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import preprocess_sequences
import utils
import neural_network
import train_model



PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spikeprot0815.fasta"
#PATH_SEQ_CLADE = PATH_PRE + "hcov_global.tsv"
HEADERS = PATH_PRE + "clade_assignment_headers.tabular"
GALAXY_CLADE_ASSIGNMENT = PATH_PRE + "clade_assignment_galaxy_0.5_Mil.tabular"
PATH_SAMPLES_CLADES = PATH_PRE + "sample_clade_sequence_df.csv"
PATH_F_DICT = PATH_PRE + "f_word_dictionaries.json"
PATH_R_DICT = PATH_PRE + "r_word_dictionaries.json"
PATH_ALL_SAMPLES_CLADES = PATH_PRE + "samples_clades.json"

PATH_TRAINING_CLADES = "data/specific_clade_in_out.json"


PRETRAIN_GEN_LOSS = "data/generated_files/pretr_gen_loss.txt"
PRETRAIN_GEN_TEST_LOSS = "data/generated_files/pretr_gen_test_loss.txt"

TRAIN_GEN_TOTAL_LOSS = "data/generated_files/tr_gen_total_loss.txt"
TRAIN_GEN_FAKE_LOSS = "data/generated_files/tr_gen_fake_loss.txt"
TRAIN_GEN_TRUE_LOSS = "data/generated_files/tr_gen_true_loss.txt"

TRAIN_DISC_TOTAL_LOSS = "data/generated_files/tr_disc_total_loss.txt"
TRAIN_DISC_FAKE_LOSS = "data/generated_files/tr_disc_fake_loss.txt"
TRAIN_DISC_TRUE_LOSS = "data/generated_files/tr_disc_true_loss.txt"

TEST_LOSS = "data/generated_files/te_loss.txt"

PRETRAIN_ENC_MODEL = "data/generated_files/pretrain_gen_encoder"
PRETRAIN_GEN_MODEL = "data/generated_files/pretrain_gen_decoder"
TRAIN_ENC_MODEL = "data/generated_files/enc_model"
TRAIN_GEN_MODEL = "data/generated_files/gen_model"
SAVE_TRUE_PRED_SEQ = "data/generated_files/true_predicted_df.csv"
TR_MUT_INDICES = "data/generated_files/tr_mut_indices.json"


l_dist_name = "levenshtein_distance"
LEN_AA = 1273
SCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# Neural network parameters
embedding_dim = 128
batch_size = 32
enc_units = 128
pretrain_epochs = 5
epochs = 3
max_l_dist = 10
train_size = 0.8
random_clade_size = 20


# https://www.tensorflow.org/text/tutorials/nmt_with_attention

def get_samples_clades():
    print("Reading clade assignments...")
    #samples_clades = preprocess_sequences.get_samples_clades(GALAXY_CLADE_ASSIGNMENT)
    samples_clades = preprocess_sequences.get_galaxy_samples_clades(GALAXY_CLADE_ASSIGNMENT)
    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq_galaxy_clades(PATH_SEQ, samples_clades)
    print(encoded_sequence_df)
    

def read_files():
    #to preprocess once, uncomment get_samples_clades
    #get_samples_clades()
    print("Preprocessing sample-clade assignment file...")
    dataf = pd.read_csv(PATH_SAMPLES_CLADES, sep=",")
    filtered_dataf = preprocess_sequences.filter_samples_clades(dataf)
    forward_dict = utils.read_json(PATH_F_DICT)
    rev_dict = utils.read_json(PATH_R_DICT)
    clades_in_clades_out = utils.read_json(PATH_TRAINING_CLADES)
    print(clades_in_clades_out)
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, filtered_dataf, train_size=train_size, edit_threshold=max_l_dist, random_size=random_clade_size)
    start_training(len(rev_dict) + 1, forward_dict, rev_dict)


def start_training(vocab_size, forward_dict, rev_dict):
    encoder, decoder = neural_network.make_generator_model(LEN_AA, vocab_size, embedding_dim, enc_units, batch_size)
    pretrain_gen_loss = list()
    pretrain_gen_test_loss = list()

    print("Loading datasets...")
    tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')

    combined_X = list()
    combined_y = list()
    combined_x_y_l = list()
    # load train data
    print("Loading training datasets...")
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        X = tr_clade_df["X"].tolist()
        y = tr_clade_df["Y"].tolist()
        combined_X.extend(X)
        combined_y.extend(y)
        print(len(X), len(y))

    combined_te_X = list()
    combined_te_y = list()
    # load test data
    print("Loading test datasets...")
    for te_name in te_clade_files:
        te_clade_df = pd.read_csv(te_name, sep="\t")
        te_X = te_clade_df["X"].tolist()
        te_y = te_clade_df["Y"].tolist()
        combined_te_X.extend(te_X)
        combined_te_y.extend(te_y)
        print(len(te_X), len(te_y))
    print()
    print("train and test data sizes")
    print(len(combined_X), len(combined_y), len(combined_te_X), len(combined_te_y))
    combined_X = np.array(combined_X)
    combined_y = np.array(combined_y)
    combined_te_X = np.array(combined_te_X)
    combined_te_y = np.array(combined_te_y)

    te_batch_size = combined_te_X.shape[0]
    print("Te batch size: {}".format(str(te_batch_size)))
    sys.exit()
    # get test dataset as sliced tensors
    test_dataset_in = tf.data.Dataset.from_tensor_slices((combined_te_X)).batch(te_batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((combined_te_y)).batch(te_batch_size)

    # create discriminator model
    disc_parent_encoder_model, disc_gen_encoder_model = neural_network.make_disc_par_gen_model(LEN_AA, vocab_size, embedding_dim, enc_units)
    discriminator = neural_network.make_discriminator_model(LEN_AA, vocab_size, embedding_dim, enc_units)

    # use the pretrained generator and train it along with discriminator
    print("Training Generator and Discriminator...")
    train_gen_total_loss = list()
    train_gen_true_loss = list()
    train_gen_fake_loss = list()
    train_disc_total_loss = list()
    train_disc_true_loss = list()
    train_disc_fake_loss = list()
    train_te_loss = list()

    X_train = combined_X
    y_train = combined_y

    n_train_batches = int(X_train.shape[0]/float(batch_size))
    print("Num of train batches: {}".format(str(n_train_batches)))
    test_data_load = [test_dataset_in, test_dataset_out]

    # balance tr data by mutations
    parent_child_mut_indices = utils.get_mutation_tr_indices(X_train, y_train, forward_dict, rev_dict)
    utils.save_as_json(TR_MUT_INDICES, parent_child_mut_indices)

    for n in range(epochs):
        print("Training epoch {}/{}...".format(str(n+1), str(epochs)))
        epo_gen_true_loss, epo_gen_fake_loss, epo_total_gen_loss, epo_disc_true_loss, epo_disc_fake_loss, epo_total_disc_loss, encoder, decoder = train_model.start_training_mut_balanced([X_train, y_train], n, encoder, decoder, disc_parent_encoder_model, disc_gen_encoder_model, discriminator, enc_units, vocab_size, n_train_batches, batch_size, test_data_load, parent_child_mut_indices)

        print("Training loss at step {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(n+1), str(epochs), str(epo_gen_true_loss), str(epo_gen_fake_loss), str(epo_total_gen_loss), str(epo_disc_true_loss), str(epo_disc_fake_loss), str(epo_total_disc_loss)))

        train_gen_total_loss.append(epo_total_gen_loss)
        train_gen_true_loss.append(epo_gen_true_loss)
        train_gen_fake_loss.append(epo_gen_fake_loss)

        train_disc_total_loss.append(epo_total_disc_loss)
        train_disc_true_loss.append(epo_disc_true_loss)
        train_disc_fake_loss.append(epo_disc_fake_loss)

        # predict seq on test data
        print("Prediction on test data...")
        with tf.device('/device:cpu:0'):
            epo_tr_gen_te_loss = utils.predict_sequence(test_dataset_in, test_dataset_out, LEN_AA, vocab_size, enc_units, TRAIN_ENC_MODEL, TRAIN_GEN_MODEL)
        train_te_loss.append(epo_tr_gen_te_loss)

    # save loss files
    np.savetxt(TRAIN_GEN_TOTAL_LOSS, train_gen_total_loss)
    np.savetxt(TRAIN_GEN_FAKE_LOSS, train_gen_fake_loss)
    np.savetxt(TRAIN_GEN_TRUE_LOSS, train_gen_true_loss)
    np.savetxt(TRAIN_DISC_FAKE_LOSS, train_disc_fake_loss)
    np.savetxt(TRAIN_DISC_TRUE_LOSS, train_disc_true_loss)
    np.savetxt(TRAIN_DISC_TOTAL_LOSS, train_disc_total_loss)
    np.savetxt(TEST_LOSS, train_te_loss)


if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
