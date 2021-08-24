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
PATH_SEQ = PATH_PRE + "spikeprot0815.fasta" #"spike_protein.fasta"
PATH_SEQ_CLADE = PATH_PRE + "hcov_global.tsv" #"ncov_global.tsv"
PATH_CLADES = "data/specific_clade_in_out.json" # "data/clade_in_clade_out.json" #  clade_in_clade_out_19A_20A.json

PRETRAIN_GEN_LOSS = "data/generated_files/pretr_gen_loss.txt"
PRETRAIN_GEN_TEST_LOSS = "data/generated_files/pretr_gen_test_loss.txt"

TRAIN_GEN_LOSS = "data/generated_files/tr_gen_loss.txt"
TRAIN_GEN_TRUE_LOSS = "data/generated_files/tr_gen_true_loss.txt"
TRAIN_DISC_LOSS = "data/generated_files/tr_disc_loss.txt"
TEST_LOSS = "data/generated_files/te_loss.txt"

PRETRAIN_ENC_MODEL = "data/generated_files/pretrain_gen_encoder"
PRETRAIN_GEN_MODEL = "data/generated_files/pretrain_gen_decoder"
TRAIN_ENC_MODEL = "data/generated_files/enc_model"
TRAIN_GEN_MODEL = "data/generated_files/gen_model"
SAVE_TRUE_PRED_SEQ = "data/generated_files/true_predicted_df.csv"
LEN_AA = 1273
SCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# Neural network parameters
embedding_dim = 128
batch_size = 32
enc_units = 128
pretrain_epochs = 5
epochs = 5
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

    start_training(len(rev_dict) + 1)


def start_training(vocab_size):
        
    encoder, decoder = neural_network.make_generator_model(LEN_AA, vocab_size, embedding_dim, enc_units, batch_size)
    pretrain_gen_loss = list()
    pretrain_gen_test_loss = list()
    
    print("Loading datasets...")
    tr_clade_files = glob.glob('data/train/*.csv')
    te_clade_files = glob.glob('data/test/*.csv')

    #train_size = 40000
    #test_size = 10000

    # load train data
    for name in tr_clade_files:
        tr_clade_df = pd.read_csv(name, sep="\t")
        X = tr_clade_df["X"]
        y = tr_clade_df["Y"]
        print(tr_clade_df.shape)

    # load test data
    for te_name in te_clade_files:
        te_clade_df = pd.read_csv(te_name, sep="\t")
        te_X = te_clade_df["X"]
        te_y = te_clade_df["Y"]
        print(te_clade_df.shape)

    #X = X[:train_size]
    #y = y[:train_size]

    #te_X = te_X[:test_size]
    #te_y = te_y[:test_size]

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
    train_gen_loss = list()
    train_gen_true_loss = list()
    train_disc_loss = list()
    train_te_loss = list()
    
    X_train = X
    y_train = y
    # get training dataset as sliced tensors
    dataset_in = tf.data.Dataset.from_tensor_slices((X_train)).batch(batch_size)
    dataset_out = tf.data.Dataset.from_tensor_slices((y_train)).batch(batch_size)
    
    n_train_batches = int(X_train.shape[0]/float(batch_size))
    print("Num of train batches: {}".format(str(n_train_batches)))
    for n in range(epochs):
        print("Training epoch {}/{}...".format(str(n+1), str(epochs)))
        epo_gen_loss, epo_disc_loss, gen_true_loss, encoder, decoder = train_model.start_training([dataset_in, dataset_out], encoder, decoder, disc_parent_encoder_model, disc_gen_encoder_model, discriminator, enc_units, vocab_size, n_train_batches)
        print("Training loss at step {}/{}: Generator true loss: {}, Generator loss: {}, Discriminator loss :{}".format(str(n+1), str(epochs), str(gen_true_loss), str(epo_gen_loss), str(epo_disc_loss)))
        train_gen_loss.append(epo_gen_loss)
        train_disc_loss.append(epo_disc_loss)
        train_gen_true_loss.append(gen_true_loss)
        # predict seq on test data
        print("Num of test batches: {}".format(str(n_test_batches)))
        print("Prediction on test data...")
        with tf.device('/device:cpu:0'):
            epo_tr_gen_te_loss = predict_sequence(test_dataset_in, test_dataset_out, seq_len, vocab_size, TRAIN_ENC_MODEL, TRAIN_GEN_MODEL)
        train_te_loss.append(epo_tr_gen_te_loss)
    
    # save loss files
    np.savetxt(TRAIN_GEN_LOSS, train_gen_loss)
    np.savetxt(TRAIN_DISC_LOSS, train_disc_loss)
    np.savetxt(TRAIN_GEN_TRUE_LOSS, train_gen_true_loss)
    np.savetxt(TEST_LOSS, train_te_loss)


def predict_sequence(test_dataset_in, test_dataset_out, seq_len, vocab_size, enc_path, dec_path):
    avg_test_loss = []
    i = 0
    loaded_encoder = tf.keras.models.load_model(enc_path)
    loaded_generator = tf.keras.models.load_model(dec_path)
    #true_x = list()
    #true_y = list()
    #predicted_y = list()
    for step, (x, y) in enumerate(zip(test_dataset_in, test_dataset_out)):
        batch_x_test = utils.convert_to_array(x)
        batch_y_test = utils.convert_to_array(y)
        batch_size = batch_x_test.shape[0]
        print(batch_x_test)
        print("Test Batch size:".format(str(batch_size)))
        if batch_x_test.shape[0] == batch_size:
            # generated noise for variation in predicted sequences
            noise = tf.random.normal((batch_size, enc_units))
            enc_output, enc_state = loaded_encoder(batch_x_test, training=False)
            # add noise to the encoder state
            enc_state = tf.math.add(enc_state, noise)
            dec_state = enc_state
            # generate seqs stepwise - teacher forcing
            generated_logits, _, loss = gen_step_predict(seq_len, batch_size, vocab_size, loaded_generator, dec_state, batch_y_test)
            # collect true_x, true_y and predicted_y into a dataframe
            #p_y = tf.math.argmax(generated_logits, axis=-1)[1]
            #one_x = utils.convert_to_string_list(batch_x_test[1])
            #one_y = utils.convert_to_string_list(batch_y_test[1])
            #pred_y = utils.convert_to_string_list(p_y)
            #true_x.append(one_x)
            #true_y.append(one_y)
            #predicted_y.append(pred_y)

            print("Test: Batch {} loss: {}".format(str(i), str(loss)))
            avg_test_loss.append(loss)
            i += 1
    #true_predicted_df = pd.DataFrame(list(zip(true_x, true_y, predicted_y)), columns=["True_X", "True_Y", "Predicted_Y"])
    #true_predicted_df.to_csv(SAVE_TRUE_PRED_SEQ, index=None)
    mean_loss = np.mean(avg_test_loss)
    print("Total test loss: {}".format(str(mean_loss)))
    return mean_loss


def gen_step_predict(seq_len, batch_size, vocab_size, gen_decoder, dec_state, real_o):
    step_loss = tf.constant(0.0)
    pred_logits = np.zeros((batch_size, seq_len, vocab_size))
    # set initial token
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(seq_len):
        o_token = real_o[:, t:t+1]
        dec_result, dec_state = gen_decoder([i_token, dec_state], training=False)
        dec_numpy = dec_result.numpy()
        pred_logits[:, t, :] = np.reshape(dec_numpy, (dec_numpy.shape[0], dec_numpy.shape[2]))
        loss = SCE(o_token, dec_result)
        step_loss += loss
        dec_tokens = tf.math.argmax(dec_result, axis=-1)
        # teacher forcing, set current output as the next input
        i_token = dec_tokens
    step_loss = step_loss / seq_len
    pred_logits = tf.convert_to_tensor(pred_logits)
    return pred_logits, gen_decoder, step_loss



if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
