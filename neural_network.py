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


class Decoder(tf.keras.layers.Layer):
  def __init__(self, output_vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.output_vocab_size = output_vocab_size
    self.embedding_dim = embedding_dim
    self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                               embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh, use_bias=False)
    self.fc = tf.keras.layers.Dense(self.output_vocab_size, use_bias=False, activation=tf.keras.activations.softmax)


  def call(self, inputs, state=None, training=False):
    vectors = self.embedding(inputs)
    rnn_output, state = self.gru(vectors, initial_state=state)
    attention_vector = self.Wc(rnn_output)
    logits = self.fc(attention_vector)
    return logits


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
    gen_output, gen_state = gen_gru(embed)
    encoder_model = tf.keras.Model([gen_inputs], [gen_output, gen_state])
    # run model
    enc_output, enc_state = encoder_model(inputs, training=True)
    
    # Create decoder for Generator
    noise = tf.keras.Input(shape=(enc_units,))
    enc_state = tf.math.add(enc_state, noise)
    new_tokens = tf.keras.Input(shape=(seq_len,))
    decoder = Decoder(vocab_size, embedding_dim, enc_units)
    # run decoder
    logits = decoder(new_tokens, state=enc_state, training=True)
    # Create GAN
    gen_model = tf.keras.Model([inputs, new_tokens, noise], [logits])
    # Save encoder's weights shared by discriminator's encoder model
    encoder_model.save_weights(ENC_WEIGHTS_SAVE_PATH)
    return gen_model, encoder_model


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
    enc_outputs, enc_state = enc_GRU(parent_inputs_embedding)
    disc_par_encoder_model = tf.keras.Model([parent_inputs], [enc_state])

    # generated seq encoder model
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    gen_enc_inputs = tf.keras.layers.Dense(embedding_dim, activation='linear', use_bias=False)(gen_inputs)
    gen_enc_outputs, gen_enc_state = enc_GRU(gen_enc_inputs)
    disc_gen_encoder_model = tf.keras.Model([gen_inputs], [gen_enc_state])

    # initialize weights of discriminator's encoder model for parent and generated seqs
    disc_par_encoder_model.load_weights(ENC_WEIGHTS_SAVE_PATH)
    disc_gen_encoder_model.layers[1].set_weights(enc_embedding.get_weights())

    return disc_par_encoder_model, disc_gen_encoder_model


def make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units):   
    input_parent = tf.keras.Input(shape=(enc_units,))
    input_generated = tf.keras.Input(shape=(enc_units,))
    
    inputs_concatenated = tf.keras.layers.Concatenate()([input_parent + input_generated])
    x = tf.keras.layers.Dropout(0.2)(inputs_concatenated)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(enc_units/2)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output_class = tf.keras.layers.Dense(1, activation='linear')(x)
    
    disc_model = tf.keras.Model([input_parent, input_generated], [output_class])
    
    return disc_model
