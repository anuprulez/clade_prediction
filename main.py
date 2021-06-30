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
#import neural_network
import sequence_to_sequence
import container_classes


PATH_PRE = "data/ncov_global/"
PATH_SEQ = PATH_PRE + "spike_protein.fasta" #"ncov_global.fasta"
PATH_SEQ_CLADE = PATH_PRE + "ncov_global.tsv"
PATH_CLADES = "data/clade_in_clade_out.json"
KMER_SIZE = 3
EMBED_DIM = 4

# https://www.tensorflow.org/text/tutorials/nmt_with_attention


def read_files():
    samples_clades = preprocess_sequences.get_samples_clades(PATH_SEQ_CLADE)
    
    clades_in_clades_out = utils.read_json(PATH_CLADES)

    print("Preprocessing sequences...")
    encoded_sequence_df, forward_dict, _ = preprocess_sequences.preprocess_seq(PATH_SEQ, samples_clades, KMER_SIZE)
    
    print("Generating cross product...")
    preprocess_sequences.make_cross_product(clades_in_clades_out, encoded_sequence_df)
    
    print("Transforming generated samples...")
    preprocess_sequences.transform_encoded_samples()
    
    print("Reading in/out sequences...")
    train_samples = preprocess_sequences.read_in_out_sequences()
    
    vocab_size, seq_len = utils.embedding_info(forward_dict, train_samples)
    
    print("Creating neural network...")

    sample_input = [[np.random.randint(vocab_size)] for i in range(seq_len)]
    sample_input = np.array(sample_input)
    #sample_input = sample_input.reshape((1, seq_len))
    print(sample_input)
    print(sample_input.shape)
    
    sample_output = [[np.random.randint(vocab_size)] for i in range(seq_len)]
    sample_output = np.array(sample_output)
    #sample_input = sample_input.reshape((1, seq_len))
    print(sample_output)
    print(sample_output.shape)
    #noise = np.zeros((seq_len, vocab_size))
    #noise[np.arange(seq_len), sample_input] = 1
    #noise = noise.reshape((1, noise.shape[0], noise.shape[1]))

    embedding_dim = 4
    units = 16
    encoder = sequence_to_sequence.Encoder(vocab_size, embedding_dim, units)
    enc_output, enc_state = encoder(sample_input)
    
    print(enc_output, enc_output.shape)
    print()
    print(enc_state, enc_state.shape)
    
    
    decoder = sequence_to_sequence.Decoder(vocab_size, embedding_dim, units)
    
    start_index = 0 #output_text_processor._index_lookup_layer('[START]').numpy()
    first_token = tf.constant([[start_index]] * sample_output.shape[0])
    print(first_token)
    
    dec_output, dec_state = decoder(
        inputs = container_classes.DecoderInput(new_tokens=first_token, enc_output=enc_output, mask=(sample_input != 0)), state = enc_state
    )

    print(dec_output.logits.shape)
    print()
    print(dec_state.shape)

if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
