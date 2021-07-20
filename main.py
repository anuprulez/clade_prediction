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
batch_size = 8
units = 16

# https://www.tensorflow.org/text/tutorials/nmt_with_attention


def read_files():
    '''samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    
    clades_in_clades_out = utils.read_json(PATH_CLADES)

    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, rev_dict = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades, KMER_SIZE)
    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df)
    
    print("Transforming generated samples...")
    preprocess_sequences.transform_encoded_samples()
    
    print("Reading in/out sequences...")
    train_samples = preprocess_sequences.read_in_out_sequences()
    
    vocab_size, seq_len = utils.embedding_info(forward_dict, train_samples)
    
    train_x = train_samples["Sequence_x"].to_numpy()
    train_y = train_samples["Sequence_y"].to_numpy()

    train_x = [list(map(int, lst)) for lst in train_x]
    train_y = [list(map(int, lst)) for lst in train_y]'''

    print("Creating neural network...")
    
    seq_len = 50
    vocab_size = 250
    
    train_x = [np.random.randint(vocab_size, size=seq_len) for i in range(3 * batch_size)]
    train_x = np.array(train_x)
    print(train_x.shape)

    train_y = [np.random.randint(vocab_size, size=seq_len) for i in range(3 * batch_size)]
    train_y = np.array(train_y)
    print(train_y.shape)
    
    dataset_in = tf.data.Dataset.from_tensor_slices((train_x)).batch(batch_size)
    dataset_out = tf.data.Dataset.from_tensor_slices((train_y)).batch(batch_size)

    start_training(dataset_in, dataset_out, embedding_dim, units, batch_size, seq_len, vocab_size)

def start_training(input_batch, output_batch, embedding_dim, units, batch_size, seq_len, vocab_size):

    '''encoder = sequence_to_sequence.Encoder(vocab_size, embedding_dim, units)
    enc_output, enc_state = encoder(sample_input)
    print("Encoder output:")
    print(enc_output.shape, enc_state.shape)
    print()

    decoder = sequence_to_sequence.Decoder(vocab_size, embedding_dim, units)

    dec_output, dec_state = decoder(
        inputs = container_classes.DecoderInput(new_tokens=input_batch, enc_output=enc_output, mask=(input_batch != 0)), state = enc_state
    )
    print("Decoder output:")
    print(dec_output.logits.shape, dec_state.shape)
    print()'''
    
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
    
    print("Start training ...")
    
    epochs = 5
    
    for n in range(epochs):
        print("Loss after training step: {}".format(str(n+1)))
        batch_learning = model.train_step([input_batch, output_batch])
        print(np.round(batch_learning["epo_loss"], 4))
        print()
    
    # generate test data
    print("Generating test data...")
    test_x = [np.random.randint(vocab_size, size=seq_len) for i in range(3 * batch_size)]
    test_x = np.array(test_x)
    print(test_x.shape)

    test_y = [np.random.randint(vocab_size, size=seq_len) for i in range(3 * batch_size)]
    test_y = np.array(test_y)
    print(test_y.shape)
    
    test_dataset_in = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)
    test_dataset_out = tf.data.Dataset.from_tensor_slices((test_y)).batch(batch_size)
    
    test_model(test_dataset_in, test_dataset_out, model)
        
def test_model(test_in, test_out, model):
    avg_test_loss = []
    for batch_x_test, batch_y_test in zip(test_in, test_out):
        #print(batch_x_test.shape, batch_y_test.shape)
        enc_output, enc_state = model.encoder(batch_x_test)
        #print("Test encoder output:")
        #print(enc_output.shape, enc_state.shape)
        #print()
        input_mask = batch_x_test != 0
        target_mask = batch_y_test != 0
        new_tokens = tf.fill([batch_size, 50], 0) # self.start_token
        decoder_input = container_classes.DecoderInput(new_tokens=new_tokens, enc_output=enc_output, mask=input_mask)
        dec_result, dec_state = model.decoder(decoder_input, state=enc_state)
        #print("Test decoder output:")
        #print(dec_result.logits.shape, dec_state.shape)

        y = batch_y_test
        y_pred = dec_result.logits
        pred_tokens = tf.argmax(y_pred, axis=-1)
        print(y.shape, y_pred.shape)
        loss = model.loss(y, y_pred)
        #print(y[0])
        #print()
        #print(pred_tokens[0])
        average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
        real_loss = average_loss.numpy()
        print("Batch test loss: {}".format(str(real_loss)))
        avg_test_loss.append(real_loss)
    print()
    print("Total test loss: {}".format(str(np.mean(avg_test_loss))))

if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
