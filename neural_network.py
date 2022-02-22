import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import h5py
from scipy.spatial import distance

import preprocess_sequences
import bahdanauAttention


GEN_ENC_WEIGHTS = "data/generated_files/generator_encoder_weights.h5"
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
DISC_DROPOUT = 0.2
RECURR_DROPOUT = 0.2
LEAKY_ALPHA = 0.1


def make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size, s_stateful):
    # Create encoder model for Generator
    # define layers
    gen_inputs = tf.keras.Input(batch_shape=(batch_size, s_stateful)) #batch_size, s_stateful
    enc_i_state_f = tf.keras.Input(shape=(enc_units,))
    enc_i_state_b = tf.keras.Input(shape=(enc_units,))
    gen_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
        embeddings_regularizer="l2", #mask_zero=True
    )

    gen_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                    kernel_regularizer="l2",
                    recurrent_regularizer="l2",
                    recurrent_initializer='glorot_normal',
                    kernel_initializer="glorot_normal",
    				return_sequences=True,
                    stateful=True,
    				return_state=True))

    embed = gen_embedding(gen_inputs)
    embed = tf.keras.layers.SpatialDropout1D(ENC_DROPOUT)(embed)
    enc_output, enc_f, enc_b = gen_gru(embed, initial_state=[enc_i_state_f, enc_i_state_b])
    encoder_model = tf.keras.Model([gen_inputs, enc_i_state_f, enc_i_state_b], [enc_output, enc_f, enc_b])

    # Create decoder for Generator
    dec_input_state = tf.keras.Input(shape=(2 * enc_units,))
    new_tokens = tf.keras.Input(shape=(1,)) # batch_size, seq_len
    # define layers
    dec_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
        embeddings_regularizer="l2", #mask_zero=True
    )

    dec_gru = tf.keras.layers.GRU(2 * enc_units,
                                   kernel_regularizer="l2",
                                   recurrent_regularizer="l2",
                                   recurrent_initializer='glorot_normal',
                                   kernel_initializer="glorot_normal",
                                   return_sequences=True,
                                   return_state=True)

    dec_dense = tf.keras.layers.Dense(enc_units, activation='relu',
        kernel_regularizer="l2",
    )

    dec_fc = tf.keras.layers.Dense(vocab_size, activation='softmax',
        kernel_regularizer="l2",
    )

    vectors = dec_embedding(new_tokens)
    vectors = tf.keras.layers.SpatialDropout1D(DEC_DROPOUT)(vectors)
    rnn_output, dec_state = dec_gru(vectors, initial_state=dec_input_state)
    rnn_output = tf.keras.layers.Dropout(DEC_DROPOUT)(rnn_output)
    rnn_output = dec_dense(rnn_output)
    rnn_output = tf.keras.layers.Dropout(DEC_DROPOUT)(rnn_output)
    logits = dec_fc(rnn_output)
    decoder_model = tf.keras.Model([new_tokens, dec_input_state], [logits, dec_state])
    encoder_model.save_weights(GEN_ENC_WEIGHTS)
    return encoder_model, decoder_model


def make_disc_par_gen_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size, s_stateful):
    # parent seq encoder model
    parent_inputs = tf.keras.Input(shape=(seq_len,)) #batch_size, s_stateful
    enc_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_regularizer="l1_l2")

    par_gen_enc_GRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                    kernel_regularizer="l2",
                    recurrent_regularizer="l2",
                    recurrent_initializer='glorot_normal',
                    kernel_initializer="glorot_normal",
    				return_sequences=True,
                    #stateful=True,
    				return_state=True))

    parent_inputs_embedding = enc_embedding(parent_inputs)
    parent_inputs_embedding = tf.keras.layers.SpatialDropout1D(ENC_DROPOUT)(parent_inputs_embedding)
    enc_out, state_f, state_b = par_gen_enc_GRU(parent_inputs_embedding)

    disc_par_encoder_model = tf.keras.Model([parent_inputs], [enc_out, state_f, state_b])

    # generated seq encoder model
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    gen_enc_inputs = tf.keras.layers.Dense(embedding_dim, use_bias=False, activation="linear")(gen_inputs)
    gen_enc_inputs = tf.keras.layers.Dropout(ENC_DROPOUT)(gen_enc_inputs)
    gen_enc_GRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                   kernel_regularizer="l2",
                    recurrent_regularizer="l2",
                    recurrent_initializer='glorot_normal',
                    kernel_initializer="glorot_normal",
    				return_sequences=True,
                    #stateful=True,
    				return_state=True))

    gen_bi_output, gen_state_f, gen_state_b = gen_enc_GRU(gen_enc_inputs)
    disc_gen_encoder_model = tf.keras.Model([gen_inputs], [gen_bi_output, gen_state_f, gen_state_b])

    # initialize weights of discriminator's encoder model for parent and generated seqs
    disc_par_encoder_model.load_weights(GEN_ENC_WEIGHTS)
    #disc_gen_encoder_model.load_weights(GEN_ENC_WEIGHTS)
    disc_gen_encoder_model.layers[1].set_weights(disc_par_encoder_model.layers[1].get_weights())
    return disc_par_encoder_model, disc_gen_encoder_model


def make_discriminator_model(enc_units):
    parent_state = tf.keras.Input(shape=(enc_units,))
    generated_state = tf.keras.Input(shape=(enc_units,))
    x = tf.keras.layers.Concatenate()([parent_state, generated_state])
    x = tf.keras.layers.Dropout(DISC_DROPOUT)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2 * enc_units)(x)
    x = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(x)
    x = tf.keras.layers.Dropout(DISC_DROPOUT)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units)(x)
    x = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(x)
    x = tf.keras.layers.Dropout(DISC_DROPOUT)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units/2)(x)
    x = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(x)
    x = tf.keras.layers.Dropout(DISC_DROPOUT)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units/4)(x)
    x = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(x)
    x = tf.keras.layers.Dropout(DISC_DROPOUT)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    output_class = tf.keras.layers.Dense(1, activation="softmax")(x)
    disc_model = tf.keras.Model([parent_state, generated_state], [output_class])
    return disc_model
