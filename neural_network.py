import time
import sys
import os

import random
import pandas as pd
import numpy as np
import logging
import tensorflow as tf

import preprocess_sequences
import sequence_to_sequence
import utils


def make_generator_model(seq_len, vocab_size, embedding_dim, enc_units, batch_size):
    
    encoder = sequence_to_sequence.Encoder(vocab_size, embedding_dim, enc_units)
    
    decoder = sequence_to_sequence.Decoder(vocab_size, embedding_dim, enc_units)
    
    inputs = tf.keras.Input(shape=(seq_len,))
    
    enc_output, enc_state = encoder(inputs, training=True)

    new_tokens = tf.fill([batch_size, seq_len], 0)
    
    logits, dec_state = decoder(new_tokens, state=enc_state, training=True)
    
    gen_model = tf.keras.Model([inputs], [logits, dec_state])
    
    return gen_model, encoder


def make_disc_par_enc_model(seq_len, vocab_size, embedding_dim, enc_units):
    # parent seq encoder model
    parent_inputs = tf.keras.Input(shape=(None,))
    enc_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    enc_GRU = tf.keras.layers.GRU(enc_units, return_state=True)
    
    parent_inputs_embedding = enc_embedding(parent_inputs)
    #enc_outputs, fwd_hPar, fwd_cPar, bwd_hPar, bwd_cPar = enc_BiGRU(parent_inputs_embedding)
    enc_outputs, enc_state = enc_GRU(parent_inputs_embedding)
    print(enc_outputs.shape, enc_state.shape)
    #state_hPar = #Concatenate()([fwd_hPar, bwd_hPar])
    #state_cPar= #Concatenate()([fwd_cPar, bwd_cPar])
    #encoder_statePar = [state_hPar, state_cPar]
    ParentEncoder_model = tf.keras.Model([parent_inputs], [enc_state])
    
    # generated seq encoder model
    gen_inputs = tf.keras.Input(shape=(None, vocab_size))
    enc_inputsGen = tf.keras.layers.Dense(embedding_dim, activation='linear', use_bias=False)(gen_inputs)
    #enc_outputsGen, fwd_hGen, fwd_cGen, bwd_hGen, bwd_cGen = enc_BiGRU(enc_inputsGen)
    enc_outputsGen, stateGen = enc_GRU(enc_inputsGen)
    #state_hGen = Concatenate()([fwd_hGen, bwd_hGen])
    #state_cGen = Concatenate()([fwd_cGen, bwd_cGen])
    encoder_stateGen = [stateGen]
    GeneratorEncoder_model = tf.keras.Model([gen_inputs], encoder_stateGen)
    
    return ParentEncoder_model, GeneratorEncoder_model


#def make_disc_gen_enc_model(seq_len, vocab_size, embedding_dim, enc_units):
    
#    return 


def make_discriminator_model(seq_len, vocab_size, embedding_dim, enc_units):

    #Load the weights for the pretrained autoencoder
    #ParentEncoder_model.load_weights('Influenza_biLSTM_encoder_model_128_4500_weights.h5')
    #GeneratorEncoder_model.layers[1].set_weights(enc_embedding.get_weights())
    
    xPar = tf.keras.Input(shape=(None, enc_units)) #ParentEncoder_model([parent_inputs])
    xGen = tf.keras.Input(shape=(None, enc_units)) #GeneratorEncoder_model([gen_inputs])
    xConcat = tf.keras.layers.Concatenate()([xPar+xGen])
    x = tf.keras.layers.Dropout(0.2)(xConcat)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    output_class = tf.keras.layers.Dense(1, activation='linear')(x)
    
    disc_model = tf.keras.Model([xPar, xGen], [output_class]) # parent_inputs, gen_inputs
    
    return disc_model

