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

import preprocess_sequences
import bahdanauAttention


GEN_ENC_WEIGHTS = "data/generated_files/generator_encoder_weights.h5"
DROPOUT = 0.25
LEAKY_ALPHA = 0.1


def make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size):
    # Create encoder model for Generator
    # define layers

    gen_inputs = tf.keras.Input(shape=(seq_len,))
    gen_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    gen_gru = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(enc_units,
                    recurrent_dropout=DROPOUT,
                    recurrent_initializer='glorot_uniform',
    				return_sequences=True,
    				return_state=True))
    gau_noise = tf.keras.layers.GaussianNoise(1.0)

    # create model
    embed = gen_embedding(gen_inputs)
    embed = tf.keras.layers.Dropout(DROPOUT)(embed)
    enc_output, f_h, f_c, b_h, b_c = gen_gru(embed)

    state_h = tf.keras.layers.Concatenate()([f_h, b_h])
    state_c = tf.keras.layers.Concatenate()([f_c, b_c])

    encoder_model = tf.keras.Model([gen_inputs], [enc_output, state_h, state_c])


    # Create decoder for Generator
    #enc_output = tf.keras.Input(shape=(seq_len, 2 * enc_units))
    i_dec_h = tf.keras.Input(shape=(2 * enc_units,))
    i_dec_c = tf.keras.Input(shape=(2 * enc_units,))
    new_tokens = tf.keras.Input(shape=(seq_len,))
    # define layers
    dec_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    dec_gru = tf.keras.layers.LSTM(2 * enc_units,
                                   recurrent_dropout=DROPOUT,
                                   recurrent_initializer='glorot_uniform',
                                   return_sequences=True,
                                   return_state=True)
    #dec_attention = bahdanauAttention.BahdanauAttention(2 * enc_units)
    dec_wc = tf.keras.layers.Dense(2 * enc_units, activation=tf.math.tanh, use_bias=False)
    dec_fc = tf.keras.layers.Dense(vocab_size, activation='softmax')
    dec_gau_noise = tf.keras.layers.GaussianNoise(1.0)

    vectors = dec_embedding(new_tokens)
    vectors = tf.keras.layers.Dropout(DROPOUT)(vectors)

    rnn_output, dec_state_h, dec_state_c = dec_gru(vectors, initial_state=[i_dec_h, i_dec_c])
    rnn_output = tf.keras.layers.Dropout(DROPOUT)(rnn_output)
    rnn_output = tf.keras.layers.BatchNormalization()(rnn_output)
    # apply attention
    # attention_weights = []
    #context_vector, attention_weights = dec_attention(query=rnn_output, value=enc_output, mask=[])
    #context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
    #attention_vector = dec_wc(context_and_rnn_output)
    #logits = dec_fc(attention_vector)
    # decoder_model = tf.keras.Model([new_tokens, enc_output, i_dec_h, i_dec_c], [logits, dec_state_h, dec_state_c, attention_weights])
    
    logits = dec_fc(rnn_output)
    decoder_model = tf.keras.Model([new_tokens, i_dec_h, i_dec_c], [logits, dec_state_h, dec_state_c])
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
    parent_inputs_embedding = enc_embedding(parent_inputs)
    parent_inputs_embedding = tf.keras.layers.Dropout(DROPOUT)(parent_inputs_embedding)
    par_bi_output = enc_GRU(parent_inputs_embedding)
    enc_outputs = par_bi_output[0]
    enc_state = tf.keras.layers.Add()([par_bi_output[1], par_bi_output[2]])
    disc_par_encoder_model = tf.keras.Model([parent_inputs], [enc_state])

    # generated seq encoder model
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    gen_enc_inputs = tf.keras.layers.Dense(embedding_dim, use_bias=False)(gen_inputs)
    gen_enc_inputs = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(gen_enc_inputs)
    gen_enc_inputs = tf.keras.layers.Dropout(DROPOUT)(gen_enc_inputs)

    gen_bi_output = enc_GRU(gen_enc_inputs)
    gen_enc_outputs = gen_bi_output[0]
    gen_enc_state = tf.keras.layers.Add()([gen_bi_output[1], gen_bi_output[2]])

    disc_gen_encoder_model = tf.keras.Model([gen_inputs], [gen_enc_state])

    # initialize weights of discriminator's encoder model for parent and generated seqs
    disc_par_encoder_model.load_weights(GEN_ENC_WEIGHTS)
    disc_gen_encoder_model.layers[1].set_weights(disc_par_encoder_model.layers[1].get_weights())
    return disc_par_encoder_model, disc_gen_encoder_model


def make_discriminator_model(enc_units):
    parent_state = tf.keras.Input(shape=(enc_units,))
    generated_state = tf.keras.Input(shape=(enc_units,))
    x = tf.keras.layers.Concatenate()([parent_state, generated_state])
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    #x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units)(x)
    x = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    #x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units/2)(x)
    x = tf.keras.layers.LeakyReLU(LEAKY_ALPHA)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    #x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output_class = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    disc_model = tf.keras.Model([parent_state, generated_state], [output_class])
    return disc_model
