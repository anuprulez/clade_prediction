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
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3
DISC_DROPOUT = 0.2
RECURR_DROPOUT = 0.25
LEAKY_ALPHA = 0.1


class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none')

  def __call__(self, y_true, y_pred):
    loss = self.loss(y_true, y_pred)
    #mask = tf.cast(y_true != 0, tf.float32)
    #loss *= mask
    return tf.reduce_sum(loss)


class ScatterEncodings(tf.keras.layers.Layer):
  def __init__(self):
    super(ScatterEncodings, self).__init__()

  def call(self, A):
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(A), 1)
    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
    # return pairwise euclidead difference matrix
    D = 1 - tf.reduce_mean(tf.sqrt(tf.maximum(na - 2*tf.matmul(A, A, False, True) + nb, 0.0))) 
    return D


def make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size, s_stateful):
    # Create encoder model for Generator
    # define layers
    gen_inputs = tf.keras.Input(shape=(seq_len,)) #batch_size, s_stateful
    gen_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
        embeddings_regularizer="l2"
    )

    gen_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                    kernel_regularizer="l2",
                    recurrent_regularizer="l2",
                    recurrent_initializer='glorot_normal',
                    kernel_initializer="glorot_normal",
    				return_sequences=True,
                    #stateful=True,
    				return_state=True))

    #enc_distance = ScatterEncodings()
    # create model
    #gen_inputs = tf.keras.layers.Dropout(ENC_DROPOUT)(gen_inputs)
    embed = gen_embedding(gen_inputs)
    embed = tf.keras.layers.SpatialDropout1D(ENC_DROPOUT)(embed)
    #embed = tf.keras.layers.LayerNormalization()(embed)
    enc_output, state_f, state_b = gen_gru(embed)

    #state_f = tf.keras.layers.LayerNormalization()(state_f)
    #state_b = tf.keras.layers.LayerNormalization()(state_b)

    encoder_model = tf.keras.Model([gen_inputs], [enc_output, state_f, state_b])

    # Create decoder for Generator
    i_dec_f = tf.keras.Input(shape=(enc_units,))
    i_dec_b = tf.keras.Input(shape=(enc_units,))
    new_tokens = tf.keras.Input(shape=(seq_len,)) # batch_size, seq_len
    # define layers
    dec_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
        embeddings_regularizer="l2"
    )

    dec_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                                   kernel_regularizer="l2",
                                   recurrent_regularizer="l2",
                                   recurrent_initializer='glorot_normal',
                                   kernel_initializer="glorot_normal",
                                   return_sequences=True,
                                   return_state=True))

    dec_fc = tf.keras.layers.Dense(vocab_size, activation='softmax',
        kernel_regularizer="l2",
    )

    vectors = dec_embedding(new_tokens)
    vectors = tf.keras.layers.SpatialDropout1D(DEC_DROPOUT)(vectors)
    #vectors = tf.keras.layers.LayerNormalization()(vectors)
    rnn_output, dec_state_f, dec_state_b = dec_gru(vectors, initial_state=[i_dec_f, i_dec_b])
    rnn_output = tf.keras.layers.Dropout(DEC_DROPOUT)(rnn_output)
    #rnn_output = tf.keras.layers.LayerNormalization()(rnn_output)
    logits = tf.keras.layers.TimeDistributed(dec_fc)(rnn_output)

    #dec_state_f = tf.keras.layers.LayerNormalization()(dec_state_f)
    #dec_state_b = tf.keras.layers.LayerNormalization()(dec_state_b)
    #logits = dec_fc(rnn_output)
    decoder_model = tf.keras.Model([new_tokens, i_dec_f, i_dec_b], [logits, dec_state_f, dec_state_b])
    encoder_model.save_weights(GEN_ENC_WEIGHTS)
    return encoder_model, decoder_model


def make_disc_par_gen_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size, s_stateful):
    # parent seq encoder model
    parent_inputs = tf.keras.Input(shape=(seq_len,)) #batch_size, s_stateful
    enc_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_regularizer="l1_l2")

    par_gen_enc_GRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                    kernel_regularizer="l2",
                    recurrent_regularizer="l2",
                    recurrent_initializer='glorot_uniform',
    				return_sequences=True,
                    #stateful=True,
    				return_state=True))

    parent_inputs_embedding = enc_embedding(parent_inputs)
    parent_inputs_embedding = tf.keras.layers.SpatialDropout1D(ENC_DROPOUT)(parent_inputs_embedding)
    #parent_inputs_embedding = tf.keras.layers.LayerNormalization()(parent_inputs_embedding)
    enc_out, state_f, state_b = par_gen_enc_GRU(parent_inputs_embedding)

    disc_par_encoder_model = tf.keras.Model([parent_inputs], [enc_out, state_f, state_b])

    # generated seq encoder model
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    gen_enc_inputs = tf.keras.layers.Dense(embedding_dim, use_bias=False, activation="linear")(gen_inputs)
    gen_enc_inputs = tf.keras.layers.Dropout(ENC_DROPOUT)(gen_enc_inputs)
    #gen_enc_inputs = tf.keras.layers.LayerNormalization()(gen_enc_inputs)
    gen_enc_GRU = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                    kernel_regularizer="l2",
                    recurrent_regularizer="l2",
                    recurrent_initializer='glorot_uniform',
    				return_sequences=True,
                    #stateful=True,
    				return_state=True))

    gen_bi_output, gen_state_f, gen_state_b = gen_enc_GRU(gen_enc_inputs)
    disc_gen_encoder_model = tf.keras.Model([gen_inputs], [gen_bi_output, gen_state_f, gen_state_b])

    # initialize weights of discriminator's encoder model for parent and generated seqs
    disc_par_encoder_model.load_weights(GEN_ENC_WEIGHTS)
    disc_gen_encoder_model.load_weights(GEN_ENC_WEIGHTS)
    #disc_gen_encoder_model.layers[0].set_weights(disc_par_encoder_model.layers[0].get_weights())
    #disc_gen_encoder_model.layers[1].set_weights(disc_par_encoder_model.layers[1].get_weights())
    return disc_par_encoder_model, disc_gen_encoder_model


def make_discriminator_model(enc_units):
    parent_state = tf.keras.Input(shape=(enc_units,))
    generated_state = tf.keras.Input(shape=(enc_units,))
    x = tf.keras.layers.Concatenate()([parent_state, generated_state])
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
    output_class = tf.keras.layers.Dense(1, activation="linear")(x)
    disc_model = tf.keras.Model([parent_state, generated_state], [output_class])
    return disc_model

'''def create_seq2seq(features_num, latent_dim):
    #features_num=5 
    #latent_dim=40

    ##
    encoder_inputs = Input(shape=(None, features_num))
    encoded = LSTM(latent_dim, return_state=False ,return_sequences=True)(encoder_inputs)
    encoded = LSTM(latent_dim, return_state=False ,return_sequences=True)(encoded)
    encoded = LSTM(latent_dim, return_state=False ,return_sequences=True)(encoded)
    encoded = LSTM(latent_dim, return_state=True)(encoded)

    encoder = Model(input=encoder_inputs, output=encoded)
    ##

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs=Input(shape=(1, features_num))
    decoder_lstm_1 = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_lstm_2 = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_lstm_3 = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_lstm_4 = LSTM(latent_dim, return_sequences=True, return_state=True)

    decoder_dense = Dense(features_num)

    all_outputs = []
    inputs = decoder_inputs


    states_1=encoder_states
    # Placeholder values:
    states_2=states_1; states_3=states_1; states_4=states_1
    ###

    for _ in range(1):
        # Run the decoder on the first timestep
        outputs_1, state_h_1, state_c_1 = decoder_lstm_1(inputs, initial_state=states_1)
        outputs_2, state_h_2, state_c_2 = decoder_lstm_2(outputs_1)
        outputs_3, state_h_3, state_c_3 = decoder_lstm_3(outputs_2)
        outputs_4, state_h_4, state_c_4 = decoder_lstm_4(outputs_3)

        # Store the current prediction (we will concatenate all predictions later)
        outputs = decoder_dense(outputs_4)
        all_outputs.append(outputs)
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states_1 = [state_h_1, state_c_1]
        states_2 = [state_h_2, state_c_2]
        states_3 = [state_h_3, state_c_3]
        states_4 = [state_h_4, state_c_4]


    for _ in range(149):
        # Run the decoder on each timestep
        outputs_1, state_h_1, state_c_1 = decoder_lstm_1(inputs, initial_state=states_1)
        outputs_2, state_h_2, state_c_2 = decoder_lstm_2(outputs_1, initial_state=states_2)
        outputs_3, state_h_3, state_c_3 = decoder_lstm_3(outputs_2, initial_state=states_3)
        outputs_4, state_h_4, state_c_4 = decoder_lstm_4(outputs_3, initial_state=states_4)

        # Store the current prediction (we will concatenate all predictions later)
        outputs = decoder_dense(outputs_4)
        all_outputs.append(outputs)
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states_1 = [state_h_1, state_c_1]
        states_2 = [state_h_2, state_c_2]
        states_3 = [state_h_3, state_c_3]
        states_4 = [state_h_4, state_c_4]


    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)   

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

    #model = load_model('pre_model.h5')'''
