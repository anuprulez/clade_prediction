import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf

import matplotlib.pyplot as plt

import preprocess_sequences
import utils
import neural_network
import sequence_to_sequence
import container_classes
import train_model
import masked_loss


PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spike_protein.fasta" #"ncov_global.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out.json"
KMER_SIZE = 3
embedding_dim = 4
batch_size = 32
units = 8
epochs = 20
seq_len = 50
vocab_size = 250

# https://www.tensorflow.org/text/tutorials/nmt_with_attention


def read_files():
    samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    
    clades_in_clades_out = utils.read_json(PATH_CLADES)

    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades, KMER_SIZE)
    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df)
    
    #print("Transforming generated samples...")
    #train_x, train_y, test_x, test_y = preprocess_sequences.transform_encoded_samples()
    
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
    '''train_x = np.array([list(map(int, lst)) for lst in train_x])
    train_y = [list(map(int, lst)) for lst in train_y]

    test_x = [list(map(int, lst)) for lst in test_x]
    test_y = [list(map(int, lst)) for lst in test_y]'''
    
    #print(train_x)
    
    #print("Reading in/out sequences...")
    # = preprocess_sequences.read_in_out_sequences()
    
    '''vocab_size, seq_len = utils.embedding_info(forward_dict, train_samples)
    
    # get train datasets
    train_x = train_samples["Sequence_x"].to_numpy()
    train_y = train_samples["Sequence_y"].to_numpy()

    train_x = [list(map(int, lst)) for lst in train_x]
    train_y = [list(map(int, lst)) for lst in train_y]
    
    print(train_x, train_y)
    
    # get test datasets
    test_x = test_samples["Sequence_x"].to_numpy()
    test_y = test_samples["Sequence_y"].to_numpy()

    test_x = [list(map(int, lst)) for lst in test_x]
    test_y = [list(map(int, lst)) for lst in test_y]
    
    print(test_x, test_y)

    print("Creating neural network...")
    
    factor = 100
    
    train_x = [np.random.randint(vocab_size, size=seq_len) for i in range(factor * batch_size)]
    train_x = np.array(train_x)
    print(train_x.shape)

    train_y = [np.random.randint(vocab_size, size=seq_len) for i in range(factor * batch_size)]
    train_y = np.array(train_y)
    print(train_y.shape)
    
    dataset_in = tf.data.Dataset.from_tensor_slices((train_x)).batch(batch_size)
    dataset_out = tf.data.Dataset.from_tensor_slices((train_y)).batch(batch_size)

    start_training(dataset_in, dataset_out, embedding_dim, units, batch_size, seq_len, vocab_size)'''

def start_training(input_batch, output_batch, embedding_dim, units, batch_size, seq_len, vocab_size):
    
    model = train_model.TrainModel(
        embedding_dim, units,
        vocab_size,
        use_tf_function=False
    )

    # Configure the loss and optimizer
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=masked_loss.MaskedLoss(),
    )
  
    te_factor = 1
    te_batch_size = 1

    print("Generating test data...")
    test_x = [np.random.randint(vocab_size, size=seq_len) for i in range(te_factor * te_batch_size)]
    test_x = np.array(test_x)

    test_y = [np.random.randint(vocab_size, size=seq_len) for i in range(te_factor * te_batch_size)]
    test_y = np.array(test_y)
    print(test_x.shape, test_y.shape)

    print("Start training ...")  

    for n in range(epochs):

        batch_learning = model.train_step([input_batch, output_batch])
        print("Training loss at step {}: {}".format(str(n+1), str(np.round(batch_learning["epo_loss"], 4))))
        predict_sequence(test_x, test_y, model, seq_len, vocab_size, te_batch_size)


def predict_sequence(test_x, test_y, model, seq_len, vocab_size, batch_size):
    avg_test_loss = []
    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)
    attention = []
    for batch_x_test, batch_y_test in zip(test_dataset_in, test_dataset_out):
        enc_output, enc_state = model.encoder(batch_x_test)
        input_mask = batch_x_test != 0
        target_mask = batch_y_test != 0
        new_tokens = tf.fill([batch_size, seq_len], 0)

        decoder_input = container_classes.DecoderInput(new_tokens=new_tokens, enc_output=enc_output, mask=input_mask)
        dec_result, dec_state = model.decoder(decoder_input, state=enc_state)
        attention.append(dec_result.attention_weights)
        # compute loss
        y = batch_y_test
        y_pred = dec_result.logits
        pred_tokens = tf.argmax(y_pred, axis=-1)
        loss = model.loss(y, y_pred)
        average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
        real_loss = average_loss.numpy()
        avg_test_loss.append(real_loss)
    attention_stack = tf.concat(attention, axis=1)
    
    print("Total test loss: {}".format(str(np.mean(avg_test_loss))))
    print()
    plot_attention(attention_stack)
    
def plot_attention(attention_stack):
    a = attention_stack[0]
    print(np.sum(a, axis=-1))
    #_ = plt.bar(range(len(a[0, :])), a[0, :])
    #plt.imshow(np.array(a), vmin=0.0)
    #plt.show()

if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
