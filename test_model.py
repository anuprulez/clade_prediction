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


import utils




PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spike_protein.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out_19A_20A.json" #"data/clade_in_clade_out.json"
TEST_LOSS = "data/generated_files/te_tforcing_loss.txt"
embedding_dim = 128
batch_size = 32
enc_units = 128
pretrain_epochs = 5
epochs = 5
LEN_AA = 1273
vocab_size = 26
seq_len = LEN_AA
SCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def test_trained_model():
    # load test data
    te_clade_files = glob.glob('data/test/*.csv')
    total_te_loss = list()
    for te_name in te_clade_files:
        te_clade_df = pd.read_csv(te_name, sep="\t")
        te_X = te_clade_df["X"]
        te_y = te_clade_df["Y"]
        print(te_clade_df.shape)
        print("Num of test batches: {}".format(str(int(te_clade_df.shape[0]/float(batch_size)))))
        print("Prediction on test data...")
        te_loss = predict_sequence(te_X, te_y, LEN_AA, vocab_size, batch_size)


def gen_step_predict(seq_len, batch_size, vocab_size, gen_decoder, dec_state, real_o):
    step_loss = tf.constant(0.0)
    pred_logits = np.zeros((batch_size, seq_len, vocab_size))
    i_token = tf.fill([batch_size, 1], 0)
    for t in tf.range(seq_len):
        o_token = real_o[:, t:t+1]
        dec_result, dec_state = gen_decoder([i_token, dec_state], training=False)
        dec_numpy = dec_result.numpy()
        pred_logits[:, t, :] = np.reshape(dec_numpy, (dec_numpy.shape[0], dec_numpy.shape[2]))
        loss = SCE(o_token, dec_result)
        step_loss += loss
        dec_tokens = tf.math.argmax(dec_result, axis=-1)
        i_token = dec_tokens
    step_loss = step_loss / seq_len
    pred_logits = tf.convert_to_tensor(pred_logits)
    print(real_o[1])
    print()
    print(tf.math.argmax(pred_logits, axis=-1)[1])
    print("------------------")
    return pred_logits, gen_decoder, step_loss


def predict_sequence(test_x, test_y, seq_len, vocab_size, batch_size):
    avg_test_loss = []
    m_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)
    i = 0
    loaded_encoder = tf.keras.models.load_model("data/generated_files/enc_model")
    loaded_generator = tf.keras.models.load_model("data/generated_files/gen_model")
    true_x = list()
    true_y = list()
    predicted_y = list()
    for step, (x, y) in enumerate(zip(test_dataset_in, test_dataset_out)):
        batch_x_test = utils.convert_to_array(x)
        batch_y_test = utils.convert_to_array(y)
        if batch_x_test.shape[0] == batch_size:
            new_tokens = tf.fill([batch_size, 1], 0)
            noise = tf.random.normal((batch_size, enc_units))
            enc_output, enc_state = loaded_encoder(batch_x_test, training=False)
            enc_state = tf.math.add(enc_state, noise)
            dec_state = enc_state
            #generated_logits, state = loaded_generator([new_tokens, enc_state], training=False)
            #loss = m_loss(batch_y_test, generated_logits)
            
            generated_logits, _, loss = gen_step_predict(seq_len, batch_size, vocab_size, loaded_generator, dec_state, batch_y_test)
            
            p_y = tf.math.argmax(generated_logits, axis=-1)[1]
            one_x = utils.convert_to_string_list(batch_x_test[1])
            one_y = utils.convert_to_string_list(batch_y_test[1])
            pred_y = utils.convert_to_string_list(p_y)

            true_x.append(one_x)
            true_y.append(one_y)
            predicted_y.append(pred_y)

            print("Test: Batch {} loss: {}".format(str(i), str(loss)))
            avg_test_loss.append(loss)
            i += 1
    true_predicted_df = pd.DataFrame(list(zip(true_x, true_y, predicted_y)), columns=["True_X", "True_Y", "Predicted_Y"])
    true_predicted_df.to_csv("data/generated_files/true_predicted_df.csv", index=None)
    mean_loss = np.mean(avg_test_loss)
    print("Total test loss: {}".format(str(mean_loss)))
    return mean_loss


if __name__ == "__main__":
    start_time = time.time()
    test_trained_model()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
