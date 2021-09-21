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
import masked_loss



PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spikeprot0815.fasta"
PATH_SEQ_CLADE = PATH_PRE + "hcov_global.tsv"
PATH_CLADES = "data/specific_clade_in_out.json"

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
epochs = 10
n_test_train_samples = 1000
seq_len = LEN_AA


# https://www.tensorflow.org/text/tutorials/nmt_with_attention


def read_files():
    samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    clades_in_clades_out = utils.read_json(PATH_CLADES)
    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades)
    print(clades_in_clades_out)    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df)
    start_training(len(rev_dict) + 1, forward_dict, rev_dict)


def start_training(vocab_size, forward_dict, rev_dict):
    encoder, decoder = neural_network.make_generator_model(LEN_AA, vocab_size, embedding_dim, enc_units, batch_size)
    pretrain_gen_loss = list()
    pretrain_gen_test_loss = list()

    print("Loading datasets...")
    tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')

    # load train data
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        X = tr_clade_df["X"]
        y = tr_clade_df["Y"]
        X_y_l = tr_clade_df[l_dist_name]

    # load test data
    for te_name in te_clade_files:
        te_clade_df = pd.read_csv(te_name, sep="\t")
        te_X = te_clade_df["X"]
        te_y = te_clade_df["Y"]

    print("train and test data sizes")
    print(X.shape, y.shape, te_X.shape, te_y.shape)

    te_batch_size = te_X.shape[0]
    print("Te batch size: {}".format(te_batch_size))
    # get test dataset as sliced tensors
    test_dataset_in = tf.data.Dataset.from_tensor_slices((te_X)).batch(te_batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((te_y)).batch(te_batch_size)

    n_test_batches = int(te_clade_df.shape[0]/float(te_batch_size))
    # divide datasets into pretrain and train sets
    '''X_pretrain, X_train, y_pretrain, y_train  = train_test_split(X, y, test_size=0.7)
    # pretrain generator
    print("Pretraining generator...")
    print(X_pretrain.shape, y_pretrain.shape, X_train.shape, y_train.shape)
    # get pretraining dataset as sliced tensors
    pretrain_dataset_in = tf.data.Dataset.from_tensor_slices((X_pretrain)).batch(batch_size)
    pretrain_dataset_out = tf.data.Dataset.from_tensor_slices((y_pretrain)).batch(batch_size)
    
    n_pretrain_batches = int(X_pretrain.shape[0]/float(batch_size))
    print("Num of pretrain batches: {}".format(str(n_pretrain_batches)))
    for i in range(pretrain_epochs):
        print("Pre training epoch {}/{}...".format(str(i+1), str(pretrain_epochs)))
        epo_pretrain_gen_loss, encoder, decoder = train_model.pretrain_generator([pretrain_dataset_in, pretrain_dataset_out], encoder, decoder, enc_units, vocab_size, n_pretrain_batches)
        print("Pre training loss at step {}/{}: Generator loss: {}".format(str(i+1), str(pretrain_epochs), str(epo_pretrain_gen_loss)))
        pretrain_gen_loss.append(epo_pretrain_gen_loss)
        print("Pretrain: predicting on test datasets...")
        print("Num of test batches: {}".format(str(n_test_batches)))
        with tf.device('/device:cpu:0'):
            epo_pt_gen_te_loss = predict_sequence(test_dataset_in, test_dataset_out, seq_len, vocab_size, PRETRAIN_ENC_MODEL, PRETRAIN_GEN_MODEL)
        pretrain_gen_test_loss.append(epo_pt_gen_te_loss)
    np.savetxt(PRETRAIN_GEN_LOSS, pretrain_gen_loss)
    np.savetxt(PRETRAIN_GEN_TEST_LOSS, pretrain_gen_test_loss)'''

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

    X_train = X
    y_train = y

    n_train_batches = int(X_train.shape[0]/float(batch_size))
    print("Num of train batches: {}".format(str(n_train_batches)))
    test_data_load = [test_dataset_in, test_dataset_out]

    # balance tr data by mutations
    parent_child_mut_indices = utils.get_mutation_tr_indices(X_train, y_train, forward_dict, rev_dict)
    utils.save_as_json(TR_MUT_INDICES, parent_child_mut_indices)

    for n in range(epochs):
        print("Training epoch {}/{}...".format(str(n+1), str(epochs)))
        epo_gen_true_loss, epo_gen_fake_loss, epo_total_gen_loss, epo_disc_true_loss, epo_disc_fake_loss, epo_total_disc_loss, encoder, decoder = train_model.start_training_mut_balanced([X_train, y_train, X_y_l], n, encoder, decoder, disc_parent_encoder_model, disc_gen_encoder_model, discriminator, enc_units, vocab_size, n_train_batches, batch_size, test_data_load, parent_child_mut_indices)

        print("Training loss at step {}/{}, G true loss: {}, G fake loss: {}, Total G loss: {}, D true loss: {}, D fake loss: {}, Total D loss: {}".format(str(n+1), str(epochs), str(epo_gen_true_loss), str(epo_gen_fake_loss), str(epo_total_gen_loss), str(epo_disc_true_loss), str(epo_disc_fake_loss), str(epo_total_disc_loss)))

        train_gen_total_loss.append(epo_total_gen_loss)
        train_gen_true_loss.append(epo_gen_true_loss)
        train_gen_fake_loss.append(epo_gen_fake_loss)

        train_disc_total_loss.append(epo_total_disc_loss)
        train_disc_true_loss.append(epo_disc_true_loss)
        train_disc_fake_loss.append(epo_disc_fake_loss)

        # predict seq on test data
        print("Num of test batches: {}".format(str(n_test_batches)))
        print("Prediction on test data...")
        with tf.device('/device:cpu:0'):
            epo_tr_gen_te_loss = utils.predict_sequence(test_dataset_in, test_dataset_out, seq_len, vocab_size, enc_units, TRAIN_ENC_MODEL, TRAIN_GEN_MODEL)
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
