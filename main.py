import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf

import preprocess_sequences
import utils
#import neural_network
import sequence_to_sequence


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

    sample_input = [np.random.randint(vocab_size) for i in range(seq_len)]
    noise = np.zeros((seq_len, vocab_size))
    noise[np.arange(seq_len), sample_input] = 1
    noise = noise.reshape((1, noise.shape[0], noise.shape[1]))

    embedding_dim = 4
    units = 16
    encoder = sequence_to_sequence.Encoder(vocab_size, embedding_dim, units)
    example_enc_output, example_enc_state = encoder(noise)
    
    
    #print("Creating train data...")
    #preprocess_sequences.generate_batches()

if __name__ == "__main__":
    start_time = time.time()
    read_files()
    end_time = time.time()
    print("Program finished in {} seconds".format(str(np.round(end_time - start_time, 2))))
