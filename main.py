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
GALAXY_CLADE_ASSIGNMENT = PATH_PRE + "clade_assignment_2.9_Mil_samples.tabular"
PATH_SAMPLES_CLADES = PATH_PRE + "sample_clade_sequence_df.csv"
PATH_F_DICT = PATH_PRE + "f_word_dictionaries.json"
PATH_R_DICT = PATH_PRE + "r_word_dictionaries.json"
PATH_TRAINING_CLADES = "data/train_clade_in_out.json"
PATH_UNRELATED_CLADES = "data/unrelated_clades.json"


PRETRAIN_GEN_LOSS = "data/generated_files/pretr_gen_loss.txt"
PRETRAIN_GEN_TEST_LOSS = "data/generated_files/pretr_gen_test_loss.txt"

TRAIN_GEN_TOTAL_LOSS = "data/generated_files/tr_gen_total_loss.txt"
TRAIN_GEN_FAKE_LOSS = "data/generated_files/tr_gen_fake_loss.txt"
TRAIN_GEN_TRUE_LOSS = "data/generated_files/tr_gen_true_loss.txt"

TRAIN_DISC_TOTAL_LOSS = "data/generated_files/tr_disc_total_loss.txt"
TRAIN_DISC_FAKE_LOSS = "data/generated_files/tr_disc_fake_loss.txt"
TRAIN_DISC_TRUE_LOSS = "data/generated_files/tr_disc_true_loss.txt"

TEST_LOSS = "data/generated_files/te_loss.txt"

PRETRAIN_GEN_ENC_MODEL = "data/generated_files/pretrain_gen_encoder"
PRETRAIN_GEN_DEC_MODEL = "data/generated_files/pretrain_gen_decoder"
TRAIN_GEN_ENC_MODEL = "data/generated_files/gen_enc_model"
TRAIN_GEN_DEC_MODEL = "data/generated_files/gen_dec_model"

SAVE_TRUE_PRED_SEQ = "data/generated_files/true_predicted_df.csv"
TR_MUT_INDICES = "data/generated_files/tr_mut_indices.json"
PRETR_MUT_INDICES = "data/generated_files/pretr_mut_indices.json"


l_dist_name = "levenshtein_distance"
LEN_AA = 1273
SCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# Neural network parameters
embedding_dim = 128
batch_size = 4
enc_units = 128
pretrain_epochs = 1
epochs = 1
max_l_dist = 10
test_train_size = 0.85
pretrain_train_size = 0.5
random_clade_size = 20
to_pretrain = False
stale_folders = ["data/generated_files/", "data/train/", "data/test/", "data/tr_unrelated/", "data/te_unrelated/"]


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
    print("Cleaning up stale folders...")
    utils.clean_up(stale_folders)
    print("Preprocessing sample-clade assignment file...")
    dataf = pd.read_csv(PATH_SAMPLES_CLADES, sep=",")
    filtered_dataf = preprocess_sequences.filter_samples_clades(dataf)
    forward_dict = utils.read_json(PATH_F_DICT)
    rev_dict = utils.read_json(PATH_R_DICT)
    clades_in_clades_out = utils.read_json(PATH_TRAINING_CLADES)
    print(clades_in_clades_out)
    unrelated_clades = utils.read_json(PATH_UNRELATED_CLADES)
    print("Generating cross product of real parent child...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, filtered_dataf, train_size=test_train_size, edit_threshold=max_l_dist, random_size=random_clade_size)
    print("Generating cross product of real sequences but not parent-child...")
    preprocess_sequences.make_cross_product(unrelated_clades, filtered_dataf, train_size=1.0, edit_threshold=max_l_dist, random_size=random_clade_size, unrelated=True)
    start_training(len(rev_dict) + 1, forward_dict, rev_dict)


def start_training(vocab_size, forward_dict, rev_dict):
    encoder, decoder = neural_network.make_generator_model(LEN_AA, vocab_size, embedding_dim, enc_units, batch_size)
    pretrain_gen_loss = list()
    pretrain_gen_test_loss = list()

    print("Loading datasets...")
    tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')
    tr_unrelated_files = glob.glob("data/tr_unrelated/*.csv")

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
    print("Loading unrelated datasets...")
    unrelated_X = list()
    unrelated_y = list()
    for tr_unrelated in tr_unrelated_files:
        unrelated_clade_df = pd.read_csv(tr_unrelated, sep="\t")
        un_X = unrelated_clade_df["X"].tolist()
        un_y = unrelated_clade_df["Y"].tolist()
        unrelated_X.extend(un_X)
        unrelated_y.extend(un_y)
        print(len(un_X), len(un_y))

    unrelated_X = np.array(unrelated_X)
    unrelated_y = np.array(unrelated_y)
    print("Unrelated data sizes")
    print(len(unrelated_X), len(unrelated_y))

    print("train and test data sizes")
    print(len(combined_X), len(combined_y), len(combined_te_X), len(combined_te_y))
    combined_X = np.array(combined_X)
    combined_y = np.array(combined_y)
    combined_te_X = np.array(combined_te_X)
    combined_te_y = np.array(combined_te_y)
    
    te_batch_size = len(combined_te_X)
    print("Te batch size: {}".format(str(te_batch_size)))

    # get test dataset as sliced tensors
    test_dataset_in = tf.data.Dataset.from_tensor_slices((combined_te_X)).batch(te_batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((combined_te_y)).batch(te_batch_size)

    # divide into pretrain and train
    if to_pretrain is False:
        X_train = combined_X
        y_train = combined_y
    else:
        X_pretrain, X_train, y_pretrain, y_train  = train_test_split(combined_X, combined_y, test_size=pretrain_train_size)
        X_pretrain = np.array(X_pretrain)
        y_pretrain = np.array(y_pretrain)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print("Train data sizes")
    print(X_train.shape, y_train.shape)

    # pretrain generator
    if to_pretrain is True:
        print("Pretrain data sizes")
        print(X_pretrain.shape, y_pretrain.shape)
        print("Pretraining generator...")
        # balance tr data by mutations
        pretr_parent_child_mut_indices = utils.get_mutation_tr_indices(X_pretrain, y_pretrain, forward_dict, rev_dict)
        utils.save_as_json(PRETR_MUT_INDICES, pretr_parent_child_mut_indices)
        # get pretraining dataset as sliced tensors
        n_pretrain_batches = int(X_pretrain.shape[0]/float(batch_size))
        print("Num of pretrain batches: {}".format(str(n_pretrain_batches)))
        for i in range(pretrain_epochs):
            print("Pre training epoch {}/{}...".format(str(i+1), str(pretrain_epochs)))
            epo_pretrain_gen_loss, encoder, decoder = train_model.pretrain_generator([X_pretrain, y_pretrain], i, encoder, decoder, enc_units, vocab_size, n_pretrain_batches, batch_size, pretr_parent_child_mut_indices, pretrain_epochs)
            print("Pre training loss at step {}/{}: Generator loss: {}".format(str(i+1), str(pretrain_epochs), str(epo_pretrain_gen_loss)))
            pretrain_gen_loss.append(epo_pretrain_gen_loss)
            print("Pretrain: predicting on test datasets...")
            with tf.device('/device:cpu:0'):
                epo_pt_gen_te_loss = utils.predict_sequence(test_dataset_in, test_dataset_out, LEN_AA, vocab_size, enc_units, PRETRAIN_GEN_ENC_MODEL, PRETRAIN_GEN_DEC_MODEL)
            pretrain_gen_test_loss.append(epo_pt_gen_te_loss)
        np.savetxt(PRETRAIN_GEN_LOSS, pretrain_gen_loss)
        np.savetxt(PRETRAIN_GEN_TEST_LOSS, pretrain_gen_test_loss)

    # GAN training
    # create discriminator model
    disc_parent_encoder_model, disc_gen_encoder_model = neural_network.make_disc_par_gen_model(LEN_AA, vocab_size, embedding_dim, enc_units)
    discriminator = neural_network.make_discriminator_model(enc_units)

    # use the pretrained generator and train it along with discriminator
    print("Training Generator and Discriminator...")
    train_gen_total_loss = list()
    train_gen_true_loss = list()
    train_gen_fake_loss = list()
    train_disc_total_loss = list()
    train_disc_true_loss = list()
    train_disc_fake_loss = list()
    train_te_loss = list()

    n_train_batches = int(X_train.shape[0]/float(batch_size))
    print("Num of train batches: {}".format(str(n_train_batches)))

    # balance tr data by mutations
    tr_parent_child_mut_indices = utils.get_mutation_tr_indices(X_train, y_train, forward_dict, rev_dict)
    utils.save_as_json(TR_MUT_INDICES, tr_parent_child_mut_indices)

    for n in range(epochs):
        print("Training epoch {}/{}...".format(str(n+1), str(epochs)))
        epo_gen_true_loss, epo_gen_fake_loss, epo_total_gen_loss, epo_disc_true_loss, epo_disc_fake_loss, epo_total_disc_loss, encoder, decoder = train_model.start_training_mut_balanced([X_train, y_train, unrelated_X, unrelated_y], n, encoder, decoder, disc_parent_encoder_model, disc_gen_encoder_model, discriminator, enc_units, vocab_size, n_train_batches, batch_size, tr_parent_child_mut_indices, epochs)

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
            epo_tr_gen_te_loss = utils.predict_sequence(test_dataset_in, test_dataset_out, LEN_AA, vocab_size, enc_units, TRAIN_GEN_ENC_MODEL, TRAIN_GEN_DEC_MODEL)
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
