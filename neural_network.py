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

ENC_WEIGHTS_SAVE_PATH = "data/generated_files/generator_encoder_weights.h5"


'''class Decoder(tf.keras.layers.Layer):
  def __init__(self, output_vocab_size, embedding_dim, dec_units, seq_len, batch_size):
    super(Decoder, self).__init__()
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.dec_units = dec_units
    self.output_vocab_size = output_vocab_size
    self.embedding_dim = embedding_dim
    self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                               embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True)

    self.Wc = tf.keras.layers.Dense(dec_units, activation='relu', use_bias=True)
    self.fc = tf.keras.layers.Dense(self.output_vocab_size, use_bias=True, activation='softmax')


  def call(self, inputs, state=None, training=False):
      vectors = self.embedding(inputs)
      #vectors = tf.keras.layers.Dropout(0.2)(vectors)
      rnn_output, state = self.gru(vectors, initial_state=state)
      #rnn_output = tf.keras.layers.Dropout(0.2)(rnn_output)
      attention_vector = self.Wc(rnn_output)
      #attention_vector = tf.keras.layers.Dropout(0.2)(attention_vector)
      logits = self.fc(attention_vector)
      return logits, state'''


def make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size):

    # Create encoder model for Generator
    # define layers
    gen_inputs = tf.keras.Input(shape=(seq_len,))
    gen_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    gen_gru = tf.keras.layers.GRU(enc_units, 
    				go_backwards=True,
    				return_sequences=True,
    				return_state=True,
    				recurrent_initializer='glorot_uniform')
    inputs = tf.keras.Input(shape=(seq_len,))
    # create model
    embed = gen_embedding(gen_inputs)
    embed = tf.keras.layers.Dropout(0.2)(embed)
    embed = tf.keras.layers.BatchNormalization()(embed)
    gen_output, gen_state = gen_gru(embed)
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
    vectors = tf.keras.layers.Dropout(0.2)(vectors)
    vectors = tf.keras.layers.BatchNormalization()(vectors)
    rnn_output, state = dec_gru(vectors, initial_state=e_state)
    rnn_output = tf.keras.layers.Dropout(0.2)(rnn_output)
    rnn_output = tf.keras.layers.BatchNormalization()(rnn_output)
    attention_vector = dec_Wc(rnn_output)
    attention_vector = tf.keras.layers.LeakyReLU(0.1)(attention_vector)
    attention_vector = tf.keras.layers.Dropout(0.2)(attention_vector)
    attention_vector = tf.keras.layers.BatchNormalization()(attention_vector)
    logits = dec_fc(attention_vector)
    decoder_model = tf.keras.Model([new_tokens, e_state], [logits, state])
    encoder_model.save_weights(ENC_WEIGHTS_SAVE_PATH)
    return encoder_model, decoder_model


def make_disc_par_gen_model(seq_len, vocab_size, embedding_dim, enc_units):
    # parent seq encoder model
    parent_inputs = tf.keras.Input(shape=(seq_len,))
    enc_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    enc_GRU = tf.keras.layers.GRU(enc_units,
                                   go_backwards=True,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    parent_inputs_embedding = enc_embedding(parent_inputs)
    parent_inputs_embedding = tf.keras.layers.Dropout(0.2)(parent_inputs_embedding)
    parent_inputs_embedding = tf.keras.layers.BatchNormalization()(parent_inputs_embedding)
    enc_outputs, enc_state = enc_GRU(parent_inputs_embedding)
    disc_par_encoder_model = tf.keras.Model([parent_inputs], [enc_state])

    # generated seq encoder model
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    gen_enc_inputs = tf.keras.layers.Dense(embedding_dim, use_bias=False)(gen_inputs)
    gen_enc_inputs = tf.keras.layers.LeakyReLU(0.1)(gen_enc_inputs)
    gen_enc_inputs = tf.keras.layers.Dropout(0.2)(gen_enc_inputs)
    gen_enc_inputs = tf.keras.layers.BatchNormalization()(gen_enc_inputs)
    gen_enc_outputs, gen_enc_state = enc_GRU(gen_enc_inputs)
    disc_gen_encoder_model = tf.keras.Model([gen_inputs], [gen_enc_state])

    # initialize weights of discriminator's encoder model for parent and generated seqs
    disc_par_encoder_model.load_weights(ENC_WEIGHTS_SAVE_PATH)
    disc_gen_encoder_model.layers[1].set_weights(enc_embedding.get_weights())

    return disc_par_encoder_model, disc_gen_encoder_model


def make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units):   
    parent_state = tf.keras.Input(shape=(enc_units,))
    generated_state = tf.keras.Input(shape=(enc_units,))
    inputs_concatenated = tf.keras.layers.Concatenate()([parent_state, generated_state])
    x = tf.keras.layers.Dropout(0.2)(inputs_concatenated)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units / 2)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output_class = tf.keras.layers.Dense(1)(x)
    disc_model = tf.keras.Model([parent_state, generated_state], [output_class])
    return disc_model
