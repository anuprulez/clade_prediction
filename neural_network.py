import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import h5py

import preprocess_sequences

GEN_ENC_WEIGHTS = "data/generated_files/generator_encoder_weights.h5"
DROPOUT = 0.3
LEAKY_ALPHA = 0.1


def make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size):

    # Create encoder model for Generator
    # define layers
    gen_inputs = tf.keras.Input(shape=(seq_len,))
    gen_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    gen_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units, 
    				go_backwards=False,
    				return_sequences=True,
    				return_state=True,
    				recurrent_initializer='glorot_uniform'), merge_mode='ave')
    g_noise = tf.keras.layers.GaussianNoise(1.0)
    inputs = tf.keras.Input(shape=(seq_len,))
    # create model
    embed = gen_embedding(gen_inputs)
    embed = tf.keras.layers.Dropout(DROPOUT)(embed)
    bi_output = gen_gru(embed)
    gen_output = bi_output[0]
    gen_state = tf.keras.layers.Add()([bi_output[1], bi_output[2]])
    #gen_output, gen_state = gen_gru(embed)
    gen_state = g_noise(gen_state)
    encoder_model = tf.keras.Model([gen_inputs], [gen_output, gen_state])

    # Create decoder for Generator
    e_state = tf.keras.Input(shape=(enc_units,))
    new_tokens = tf.keras.Input(shape=(seq_len,))
    # define layers
    dec_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    dec_gru = tf.keras.layers.GRU(enc_units,
                                   return_sequences=True,
                                   return_state=True)

    dec_Wc = tf.keras.layers.Dense(enc_units, use_bias=True)
    dec_fc = tf.keras.layers.Dense(vocab_size, use_bias=True, activation='softmax')

    vectors = dec_embedding(new_tokens)
    vectors = tf.keras.layers.Dropout(DROPOUT)(vectors)
    rnn_output, state = dec_gru(vectors, initial_state=e_state)
    rnn_output = tf.keras.layers.Dropout(DROPOUT)(rnn_output)
    attention_vector = dec_Wc(rnn_output)
    attention_vector = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(attention_vector)
    attention_vector = tf.keras.layers.Dropout(DROPOUT)(attention_vector)
    logits = dec_fc(attention_vector)
    decoder_model = tf.keras.Model([new_tokens, e_state], [logits, state])
    encoder_model.save_weights(GEN_ENC_WEIGHTS)
    return encoder_model, decoder_model


def make_disc_par_gen_model(seq_len, vocab_size, embedding_dim, enc_units):
    # parent seq encoder model
    parent_inputs = tf.keras.Input(shape=(seq_len,))
    enc_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    enc_GRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                                   go_backwards=False,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform'), merge_mode='ave')
    g_noise = tf.keras.layers.GaussianNoise(1.0)
    parent_inputs_embedding = enc_embedding(parent_inputs)
    parent_inputs_embedding = tf.keras.layers.Dropout(DROPOUT)(parent_inputs_embedding)
    #enc_outputs, enc_state = enc_GRU(parent_inputs_embedding)
    par_bi_output = enc_GRU(parent_inputs_embedding)
    enc_outputs = par_bi_output[0]
    enc_state = tf.keras.layers.Add()([par_bi_output[1], par_bi_output[2]])
    enc_state = g_noise(enc_state)
    disc_par_encoder_model = tf.keras.Model([parent_inputs], [enc_state])

    # generated seq encoder model
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    gen_enc_inputs = tf.keras.layers.Dense(embedding_dim, use_bias=False)(gen_inputs)
    gen_enc_inputs = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(gen_enc_inputs)
    gen_enc_inputs = tf.keras.layers.Dropout(DROPOUT)(gen_enc_inputs)
    g_noise = tf.keras.layers.GaussianNoise(1.0)

    #gen_enc_outputs, gen_enc_state = enc_GRU(gen_enc_inputs)
    gen_bi_output = enc_GRU(gen_enc_inputs)
    gen_enc_outputs = gen_bi_output[0]
    gen_enc_state = tf.keras.layers.Add()([gen_bi_output[1], gen_bi_output[2]])
    gen_enc_state = g_noise(gen_enc_state)
    disc_gen_encoder_model = tf.keras.Model([gen_inputs], [gen_enc_state])

    # initialize weights of discriminator's encoder model for parent and generated seqs
    disc_par_encoder_model.load_weights(GEN_ENC_WEIGHTS)
    disc_gen_encoder_model.layers[1].set_weights(enc_embedding.get_weights())
    return disc_par_encoder_model, disc_gen_encoder_model


def make_discriminator_model(enc_units):
    parent_state = tf.keras.Input(shape=(enc_units,))
    generated_state = tf.keras.Input(shape=(enc_units,))
    x = tf.keras.layers.Concatenate()([parent_state, generated_state])
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    #x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units/2)(x)
    x = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    #x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units/4)(x)
    x = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    #x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output_class = tf.keras.layers.Dense(1, activation="linear")(x)
    disc_model = tf.keras.Model([parent_state, generated_state], [output_class])
    return disc_model
