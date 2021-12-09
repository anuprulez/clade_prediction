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
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
RECURR_DROPOUT = 0.25
LEAKY_ALPHA = 0.3


class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none')

  def __call__(self, y_true, y_pred):
    #shape_checker = ShapeChecker()
    #shape_checker(y_true, ('batch', 't'))
    #shape_checker(y_pred, ('batch', 't', 'logits'))

    # Calculate the loss for each item in the batch.
    loss = self.loss(y_true, y_pred)
    #shape_checker(loss, ('batch', 't'))

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, tf.float32)
    #shape_checker(mask, ('batch', 't'))
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)


def make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size, s_stateful):
    # Create encoder model for Generator
    # define layers
    gen_inputs = tf.keras.Input(batch_shape=(batch_size, s_stateful))
    gen_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
    conv1d = tf.keras.layers.Conv1D(filters=16, kernel_size=10, strides=3, activation='relu')
    gen_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                    recurrent_initializer='glorot_uniform',
    				return_sequences=True,
                    stateful=True,
    				return_state=True))

    # create model
    embed = gen_embedding(gen_inputs)
    embed = tf.keras.layers.Dropout(ENC_DROPOUT)(embed)
    #print(embed.shape)
    #conv_embed = conv1d(embed)
    #print(conv_embed.shape)
    #conv_embed = tf.keras.layers.Dropout(ENC_DROPOUT)(conv_embed)
    enc_output, state_f, state_b = gen_gru(embed)
    encoder_model = tf.keras.Model([gen_inputs], [enc_output, state_f, state_b])

    # Create decoder for Generator
    #enc_output = tf.keras.Input(shape=(seq_len, 2 * enc_units))
    i_dec_f = tf.keras.Input(shape=(enc_units,))
    i_dec_b = tf.keras.Input(shape=(enc_units,))
    new_tokens = tf.keras.Input(batch_shape=(batch_size, seq_len))
    # define layers
    dec_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
    #dec_conv1d = tf.keras.layers.Conv1D(filters=16, kernel_size=4, activation='relu')
    dec_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                                   recurrent_initializer='glorot_uniform',
                                   return_sequences=True,
                                   return_state=True))

    dec_fc = tf.keras.layers.Dense(vocab_size, activation='softmax')

    vectors = dec_embedding(new_tokens)
    vectors = tf.keras.layers.Dropout(DEC_DROPOUT)(vectors)

    #conv_dec_vectors = dec_conv1d(vectors)
    #conv_dec_vectors = tf.keras.layers.Dropout(DEC_DROPOUT)(conv_dec_vectors)
    
    rnn_output, dec_state_f, dec_state_b = dec_gru(vectors, initial_state=[i_dec_f, i_dec_b])
    rnn_output = tf.keras.layers.Dropout(DEC_DROPOUT)(rnn_output)

    logits = dec_fc(rnn_output)
    decoder_model = tf.keras.Model([new_tokens, i_dec_f, i_dec_b], [logits, dec_state_f, dec_state_b])
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
